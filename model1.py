# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Top-down model
==============

This script gives an example on how to use Pyro forecast module to backtest and
make a submission for M5 accuracy competition.

Using the top-down approach in [1], we first construct a model to predict
the aggregated sales across all items. Then we will distribute the aggregated
prediction to each product based on its total sales during the last 28 days.

The model we use is a slight modification of the heavy-tailed model at [2]. We
recommend to read that tutorial first for more explanation.

The result will barely beat all benchmark models from the competition.

**References**

    1. Rob J Hyndman and George Athanasopoulos (2018), "Forecasting: Principles and Practice",
       (https://otexts.com/fpp2/top-down.html)
    2. http://pyro.ai/examples/forecasting_ii.html#Heavy-tailed-modeling-with-LinearHMM
"""

import argparse
import os
import pickle

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.contrib.forecast import ForecastingModel, Forecaster
from pyro.infer.reparam import LinearHMMReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat

from evaluate import m5_backtest
from util import M5Data


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("BART_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


class Model(ForecastingModel):
    def model(self, zero_data, covariates):
        assert zero_data.size(-1) == 1  # univariate
        duration = zero_data.size(-2)
        feature_dim = covariates.size(-1)
        day_of_week_plate = pyro.plate("day_of_week", 7, dim=-1)

        bias = pyro.sample("bias", dist.Normal(0, 10))
        # we will use the first covariate, i.e. time, to compute trend
        trend_coef = pyro.sample("trend", dist.LogNormal(-2, 1))
        trend = trend_coef * covariates[:, 0]
        # set prior of weights of the remaining covariates
        weight = pyro.sample("weight",
                             dist.Normal(0, 1).expand([feature_dim - 1]).to_event(1))
        # encode weekly seasonality
        with day_of_week_plate:
            seasonal = pyro.sample("seasonal", dist.Normal(0, 5))
        seasonal = periodic_repeat(seasonal, duration, dim=-1)

        prediction = bias + trend + seasonal + (weight * covariates[:, 1:]).sum(-1)
        prediction = prediction.unsqueeze(-1)

        # heavy-tailed modeling for the noise
        init_dist = dist.Normal(0, 10).expand([1]).to_event(1)
        timescale = pyro.sample("timescale", dist.LogNormal(0, 1))
        trans_matrix = torch.exp(-1 / timescale)[..., None, None]
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(0, 1))
        trans_dist = dist.Normal(0, trans_scale.unsqueeze(-1)).to_event(1)
        obs_matrix = torch.tensor([[1.]])
        with day_of_week_plate:
            obs_scale = pyro.sample("obs_scale", dist.LogNormal(-2, 1))
        obs_scale = periodic_repeat(obs_scale, duration, dim=-1)

        stability = pyro.sample("stability", dist.Uniform(1, 2).expand([1]).to_event(1))
        skew = pyro.sample("skew", dist.Uniform(-1, 1).expand([1]).to_event(1))
        trans_dist = dist.Stable(stability, 0, trans_scale.unsqueeze(-1)).to_event(1)
        obs_dist = dist.Stable(stability, skew, obs_scale.unsqueeze(-1)).to_event(1)
        noise_dist = dist.LinearHMM(init_dist, trans_matrix, trans_dist,
                                    obs_matrix, obs_dist, duration=duration)

        rep = LinearHMMReparam(None, SymmetricStableReparam(), StableReparam())
        with poutine.reparam(config={"residual": rep}):
            self.predict(noise_dist, prediction)


def main(args):
    m5 = M5Data()
    # get aggregated sales of all items from all Walmart stores
    data = m5.get_aggregated_sales(m5.aggregation_levels[0])[0].unsqueeze(-1)
    # apply log transform to scale down the data
    data = data.log()

    # we create covariates from dummy time features and special events
    T0 = 0                   # begining
    T2 = data.size(-2) + 28  # end + submission-interval
    time = torch.arange(T0, float(T2), device="cpu") / 365
    covariates = torch.cat([time.unsqueeze(-1),
                            m5.get_dummy_day_of_month()[T0:T2],
                            m5.get_dummy_month_of_year()[T0:T2],
                            m5.get_event()[T0:T2],
                            m5.get_christmas()[T0:T2],
                            ], dim=-1)

    if args.cuda:
        data = data.cuda()
        covariates = covariates.cuda()

    forecaster_options = {
        "learning_rate": args.learning_rate,
        "clip_norm": args.clip_norm,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
    }

    def transform(pred, truth):
        return pred.exp(), truth.exp()

    if args.submit:
        pyro.set_rng_seed(args.seed)
        forecaster = Forecaster(Model(), data, covariates[:-28], **forecaster_options)
        samples = forecaster(data, covariates, num_samples=1000).exp()
        pred = samples.median(0).values.squeeze(-1).cpu()

        # we use top-down approach to distribute the aggregated forecast sales `pred`
        # for each items at the bottom level;
        # the proportion is calculated based on the proportion of total sales of each time
        # during the last 28 days (this follows M5 guide's benchmark models)
        sales_last28 = m5.get_aggregated_sales(**m5.aggregation_levels[-1])[:, -28:]
        proportion = sales_last28.sum(-1) / sales_last28.sum()
        prediction = proportion.ger(pred)
        m5.make_accuracy_submission(args.output_file, prediction)
    else:
        min_train_window = data.size(-2) - args.test_window - (args.num_windows - 1) * args.stride
        windows = m5_backtest(data, covariates[:-28], Model,
                              transform=transform,
                              min_train_window=min_train_window,
                              test_window=args.test_window,
                              stride=args.stride,
                              forecaster_options=forecaster_options,
                              seed=args.seed)

        with open(args.output_file, "wb") as f:
            pickle.dump(windows, f)

        for name in ["ws_rmse", "ws_pl"]:
            values = torch.tensor([w[name] for w in windows])
            print("{} = {:0.3g} +- {:0.2g}".format(name, values.mean(), values.std()))


if __name__ == "__main__":
    assert pyro.__version__ >= "1.3.0"
    parser = argparse.ArgumentParser(description="Univariate M5 daily forecasting")
    parser.add_argument("--num-windows", default=10, type=int)
    parser.add_argument("--test-window", default=28, type=int)
    parser.add_argument("-s", "--stride", default=35, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--clip-norm", default=10., type=float)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("-o", "--output-file", default="", type=str)
    parser.add_argument("--submit", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.output_file == "":
        args.output_file = os.path.join(
            RESULTS, os.path.basename(__file__)[:-3] + ".csv" if args.submit else ".pkl")

    main(args)
