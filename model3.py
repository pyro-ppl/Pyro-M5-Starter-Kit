# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Middle-out Model
================

This example model illustrates how to use M5Data class to aggregate,
disaggregate data or prediction at each aggregation levels.

Using the middle-out approach in [1], we first construct a model to predict
the aggregated sales across stores and departments. Then we will distribute
the aggregated prediction to each product based on its total sales during
the last 28 days.

**References**

    1. Rob J Hyndman and George Athanasopoulos (2018), "Forecasting: Principles and Practice",
       (https://otexts.com/fpp2/middle-out.html)
"""

import argparse
import os
import pickle

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.contrib.forecast import ForecastingModel, Forecaster
from pyro.ops.tensor_utils import periodic_repeat, periodic_features

from evaluate import get_metric_scale, m5_backtest
from util import M5Data


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("PYRO_M5_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


# We will use a model similar to the one defined in `model1.py`. But instead of working
# with the top level data, we will use 70 timeseries at store+department level here.
class Model(ForecastingModel):
    def model(self, zero_data, covariates):
        num_stores, num_depts, duration, one = zero_data.shape
        time, feature = covariates[..., :1], covariates[..., 1:]

        store_plate = pyro.plate("store", num_stores, dim=-3)
        dept_plate = pyro.plate("dept", num_depts, dim=-2)
        day_of_week_plate = pyro.plate("day_of_week", 7, dim=-1)

        with dept_plate, store_plate:
            bias = pyro.sample("bias", dist.Normal(0, 10).expand([1]).to_event(1))
            trend_coef = pyro.sample("trend", dist.LogNormal(-1, 1).expand([1]).to_event(1))
            trend = trend_coef * time

            # set prior of weights of the remaining covariates
            weight = pyro.sample("weight",
                                 dist.Normal(0, 1).expand([1, feature.size(-1)]).to_event(2))
            regressor = weight.matmul(feature.unsqueeze(-1)).squeeze(-1)

            # encode weekly seasonality
            with day_of_week_plate:
                seasonal = pyro.sample("seasonal", dist.Normal(0, 1).expand([1]).to_event(1))
            seasonal = periodic_repeat(seasonal, duration, dim=-2)

            noise_scale = pyro.sample("noise_scale", dist.LogNormal(-1, 1).expand([1]).to_event(1))

        prediction = bias + trend + seasonal + regressor
        dof = pyro.sample("dof", dist.Uniform(1, 10).expand([1]).to_event(1))
        noise_dist = dist.StudentT(dof, zero_data, noise_scale)
        self.predict(noise_dist, prediction)


def main(args):
    m5 = M5Data()
    # get aggregated sales at store+dept level
    level = ["store_id", "dept_id"]
    data = m5.get_aggregated_sales(level)
    data = data.reshape(m5.num_stores, m5.num_depts, m5.num_train_days, 1)
    # each department at each store has different scales; instead of
    # using log transform as in `model1.py`, we will scale those timeseries down
    # using the same scale of WSPL evaluation metric. Please refer to
    # evaluation section of M5 guideline for more information about scale.
    scale = get_metric_scale("pl", data).unsqueeze(-1).unsqueeze(-1)
    data = data / scale

    T0 = 0
    T2 = data.size(-2) + 28  # end + submission-interval
    T1 = T2 - 28
    time = torch.arange(float(T2), device="cpu") / 365
    covariates = torch.cat([
        time.unsqueeze(-1),
        # The follow code creates yearly features so we can learn yearly
        # pattern from the dataset.
        periodic_features(T2, 365.25, 7)
    ], dim=-1)

    if args.cuda:
        data = data.cuda()
        covariates = covariates.cuda()
        scale = scale.cuda()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    forecaster_options = {
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "clip_norm": args.clip_norm,
        "num_steps": args.num_steps,
        "log_every": args.log_every,
    }

    def transform(pred, truth):
        # note that our pred/truth are results at store+dept level;
        # we need to aggregate them to higher aggregation levels
        # to verify if the result at higher aggregative levels is still good

        pred, truth = (pred.clamp(min=0) * scale).cpu(), (truth * scale).cpu()
        num_samples, duration = pred.size(0), pred.size(-2)
        # Note that the method `m5.aggregate_samples` only aggregates non-aggregated
        # timeseries to higher levels, but we have timeseries at the middle store+dept
        # level. So we assume the all items in each department at each store have the
        # same sales, disaggregate the timeseries at store+dept level to the
        # non-aggregated level. Then aggretates the result to any higher aggregation level.
        num_items_by_dept = torch.tensor(m5.num_items_by_dept, device="cpu")
        pred = pred / num_items_by_dept.unsqueeze(-1).unsqueeze(-1)
        non_agg_pred = pred.repeat_interleave(num_items_by_dept, dim=-3)
        non_agg_pred = non_agg_pred.reshape(num_samples, -1, duration)
        # note that store+dept is the forth level from bottom up.
        agg_pred = m5.aggregate_samples(non_agg_pred, *m5.aggregation_levels[:-3])

        # similarly, we apply the same procedure for truth data
        truth = truth / num_items_by_dept.unsqueeze(-1).unsqueeze(-1)
        non_agg_truth = truth.repeat_interleave(num_items_by_dept, dim=-3)
        non_agg_truth = non_agg_truth.reshape(1, -1, duration)
        agg_truth = m5.aggregate_samples(non_agg_truth, *m5.aggregation_levels[:-3]).squeeze(0)
        return agg_pred.unsqueeze(-1), agg_truth.unsqueeze(-1)

    if args.submit:
        pyro.set_rng_seed(args.seed)
        print("Training...")
        forecaster = Forecaster(Model(), data[..., T0:T1, :], covariates[T0:T1],
                                **forecaster_options)

        print("Forecasting...")
        samples = forecaster(data[..., T0:T1, :], covariates[T0:T2],
                             num_samples=1000, batch_size=10)
        samples = samples.clamp(min=0) * scale

        # Compute the ratio of prediction w.r.t. the sales of last 28 days
        # first, we get the total sales of each department at each store in the last 28 days
        dept_store_sales = m5.get_aggregated_sales(level)[:, -28:].sum(-1)
        dept_store_sales = dept_store_sales.reshape(m5.num_stores, m5.num_depts)
        # num_items_by_dept tells us how many items in each department
        num_items_by_dept = torch.tensor(m5.num_items_by_dept)
        dept_store_sales = dept_store_sales.repeat_interleave(num_items_by_dept, dim=-1)
        # get the sales at the lowest level: store+item (this is the non-aggregated level)
        sales_last28 = m5.get_aggregated_sales(["store_id", "item_id"])[:, -28:].sum(-1)
        proportion = sales_last28 / dept_store_sales.reshape(-1)

        # after calculate the ratio, we disaggregate prediction to the bottom level
        samples = samples.squeeze(-1).repeat_interleave(num_items_by_dept, dim=-2)
        samples = samples.reshape(-1, m5.num_timeseries, 28)
        non_agg_samples = torch.poisson(samples * proportion.unsqueeze(-1))
        # aggregate the result to all aggregation levels
        agg_samples = m5.aggregate_samples(non_agg_samples, *m5.aggregation_levels)
        # cast to numpy because pyro quantile implementation is memory hungry
        print("Calculate quantiles...")
        q = np.quantile(agg_samples.numpy(), m5.quantiles, axis=0)
        print("Make uncertainty submission...")
        filename, ext = os.path.splitext(args.output_file)
        m5.make_uncertainty_submission(filename + "_uncertainty" + ext, q, float_format='%.3f')
    else:
        # calculate weight of each timeseries higher or equal to store+department level
        weight = m5.get_aggregated_ma_dollar_sales(m5.aggregation_levels[-1]).cpu()
        weight = weight / weight.sum(0, keepdim=True)
        agg_weight = m5.aggregate_samples(weight.unsqueeze(0), *m5.aggregation_levels[:-3])
        agg_weight = agg_weight.squeeze(0)

        min_train_window = T1 - T0 - args.test_window - (args.num_windows - 1) * args.stride
        print("Backtesting with skip window {}...".format(T0))
        windows = m5_backtest(data, covariates[:T1], Model,
                              weight=agg_weight,
                              skip_window=T0,
                              transform=transform,
                              min_train_window=min_train_window,
                              test_window=args.test_window,
                              stride=args.stride,
                              forecaster_options=forecaster_options,
                              num_samples=1000,
                              batch_size=10,
                              seed=args.seed)

        with open(args.output_file, "wb") as f:
            pickle.dump(windows, f)

        for name in ["ws_rmse", "ws_pl"]:
            values = torch.tensor([w[name] for w in windows])
            print("{} = {:0.3g} +- {:0.2g}".format(name, values.mean(), values.std()))


if __name__ == "__main__":
    assert pyro.__version__ >= "1.3.0"
    parser = argparse.ArgumentParser(description="Univariate M5 daily forecasting")
    parser.add_argument("--num-windows", default=3, type=int)
    parser.add_argument("--test-window", default=28, type=int)
    parser.add_argument("-s", "--stride", default=35, type=int)
    parser.add_argument("-n", "--num-steps", default=2001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--clip-norm", default=10., type=float)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("-o", "--output-file", default="", type=str)
    parser.add_argument("--submit", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.output_file == "":
        args.output_file = os.path.basename(__file__)[:-3] + (".csv" if args.submit else ".pkl")
    args.output_file = os.path.join(RESULTS, args.output_file)

    main(args)
