# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Hierarchial model
=================
"""


import argparse
import math
import os
import pickle

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.forecast import ForecastingModel, Forecaster
from pyro.nn import PyroModule, PyroParam
from pyro.ops.tensor_utils import periodic_repeat

from evaluate import eval_mae, eval_rmse, eval_pl, m5_backtest
from util import M5Data


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS = os.environ.get("PYRO_M5_RESULTS", RESULTS)
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)


def bounded_exp(x, bound=1e3):
    # this utility is very helpful at early training phase
    return (x - math.log(bound)).sigmoid() * bound


class Model(ForecastingModel):
    # it is expensive to merge covariates into the shape `10 x 3049 x duration x feature_dim`,
    # especially when many covariates have shape `duration x d`;
    # hence we store those extra covariates in the constructor.
    def __init__(self, snap, dept, saled, log_ma):
        super().__init__()
        # boolean indicating if SNAP is available
        assert snap.shape == (10, 1, snap.size(2), 1)
        self.snap = snap
        # one-hot encoding for the department of each product
        assert dept.shape == (10, 3049, 1, 7)
        self.dept = dept
        # boolean indicating if the product is available to sale in that day,
        # this is useful to mask out not-yet-available items at early days
        assert saled.shape == (10, 3049, snap.size(2), 1)
        self.saled = saled
        # moving average features, we use this covariate as a time-local feature
        assert log_ma.shape == (10, 3049, log_ma.size(2), 3)
        self.log_ma = log_ma

    def model(self, zero_data, covariates):
        assert zero_data.size(-1) == 1  # univariate
        num_stores, num_products, duration, one = zero_data.shape
        time_index = covariates.squeeze(-1)

        store_plate = pyro.plate("store", num_stores, dim=-3)
        product_plate = pyro.plate("product", num_products, dim=-2)
        day_of_week_plate = pyro.plate("day_of_week", 7, dim=-1)

        snap = self.snap[..., time_index, :]
        # subsample the data
        with product_plate:
            dept = pyro.subsample(self.dept, event_dim=1)
            saled = pyro.subsample(self.saled, event_dim=1)[..., time_index, :]
            log_ma = pyro.subsample(self.log_ma, event_dim=1)[..., time_index, :]

        # we construct latent variables for each store and each department;
        # here, we declare department dimension as event dimension for simplicity,
        # (nb: the numbers of products in each department are different)
        # the last event dimension is used to model mean/scale separately.
        with store_plate:
            ma_weight = pyro.sample("ma_weight",
                                    dist.Normal(0, 1).expand([2, log_ma.size(-1), 7]).to_event(3))
            ma_weight = ma_weight.matmul(dept.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
            moving_average = ma_weight.matmul(log_ma.unsqueeze(-1)).squeeze(-1)

            snap_weight = pyro.sample("snap_weight", dist.Normal(0, 1).expand([2, 7]).to_event(2))
            snap_weight = snap_weight.matmul(dept.unsqueeze(-1)).squeeze(-1)
            snap_effect = snap_weight * snap

            with day_of_week_plate:
                seasonal = pyro.sample("seasonal", dist.Normal(0, 1).expand([2, 7]).to_event(2))
            seasonal = seasonal.matmul(dept.unsqueeze(-1)).squeeze(-1)
            seasonal = periodic_repeat(seasonal, duration, dim=-2)

        prediction = moving_average + snap_effect + seasonal
        log_mean, log_scale = prediction[..., :1], prediction[..., 1:]
        # we add a pretty small bias 1e-3 to avoid the case mean=scale=0
        # either when saled == 0 or saled == 1
        mean = bounded_exp(log_mean) * saled + 1e-3
        scale = bounded_exp(log_scale) * saled + 1e-3

        rate = scale.reciprocal()
        concentration = mean * rate
        # alternative: GammaPoisson (or NegativeBinomial, ZeroInflatedNegativeBinomial)
        noise_dist = dist.Gamma(concentration, rate)

        with store_plate, product_plate:
            self.predict(noise_dist, mean.new_zeros(mean.shape))


class NormalGuide(PyroModule):
    def __init__(self, create_plates=None):
        super().__init__()
        self.ma_weight_loc = PyroParam(torch.zeros(10, 1, 1, 2, 3, 7), event_dim=3)
        self.ma_weight_scale = PyroParam(torch.ones(10, 1, 1, 2, 3, 7) * 0.1,
                                         dist.constraints.positive, event_dim=3)
        self.snap_weight_loc = PyroParam(torch.zeros(10, 1, 1, 2, 7), event_dim=2)
        self.snap_weight_scale = PyroParam(torch.ones(10, 1, 1, 2, 7) * 0.1,
                                           dist.constraints.positive, event_dim=2)
        self.seasonal_loc = PyroParam(torch.zeros(10, 1, 7, 2, 7), event_dim=2)
        self.seasonal_scale = PyroParam(torch.ones(10, 1, 7, 2, 7) * 0.1,
                                        dist.constraints.positive, event_dim=2)
        self.create_plates = create_plates

    def forward(self, data, covariates):
        num_stores = data.size(0)
        if self.create_plates is not None:
            product_plate = self.create_plates(data, covariates)  # noqa: F841
        store_plate = pyro.plate("store", num_stores, dim=-3)
        day_of_week_plate = pyro.plate("day_of_week", 7, dim=-1)

        with store_plate:
            pyro.sample("ma_weight",
                        dist.Normal(self.ma_weight_loc, self.ma_weight_scale).to_event(3))
            pyro.sample("snap_weight",
                        dist.Normal(self.snap_weight_loc, self.snap_weight_scale).to_event(2))
            with day_of_week_plate:
                pyro.sample("seasonal",
                            dist.Normal(self.seasonal_loc, self.seasonal_scale).to_event(2))


def create_plates(zero_data, covariates):
    # NB: with size=60, it took about 50 epochs to walk through the whole dataset
    return pyro.plate("product", zero_data.shape[1], subsample_size=60, dim=-2)


# forecasting requires too much memory resources, so we will draw samples
# in batches and cast each batch to CPU;
# in addition, we will skip the unnecessary training data and
# the corresponding covariates because our model does not require them
# to make predictions.
class M5Forecaster(Forecaster):
    def forward(self, data, covariates, num_samples, batch_size=None):
        if batch_size is not None:
            batches = []
            while num_samples > 0:
                batch = self.forward(data, covariates, min(num_samples, batch_size))
                batches.append(batch)
                num_samples -= batch_size
            return torch.cat(batches)

        # make sure the skip part has no conflict with weekly seasonal pattern
        skip = data.size(-2) // 7 * 7
        return super().forward(data[..., skip:, :], covariates[skip:], num_samples).cpu()


def main(args):
    print("Preparation...")
    m5 = M5Data()
    # get non-aggregated sales of all items from all Walmart stores
    data = m5.get_aggregated_sales(m5.aggregation_levels[-1])
    # reshape into num_stores x num_products x duration x 1
    data = data.reshape(10, 3049, -1, 1)

    T0 = 37 + 28 * 3         # begining, skip a small period to calculate moving average
    T2 = data.size(-2) + 28  # end + submission-interval
    T1 = T2 - 28             # train/test split
    assert (T2 - T0) % 28 == 0

    covariates = torch.arange(T2).unsqueeze(-1)
    # extra covariates
    snap = m5.get_snap().repeat_interleave(torch.tensor([4, 3, 3]), dim=-1)
    snap = snap.t().unsqueeze(1).unsqueeze(-1)
    dept = m5.get_dummy_dept().reshape(10, -1, 7).unsqueeze(-2)
    saled = (m5.get_prices() != 0).type(torch.get_default_dtype()).reshape(10, 3049, -1, 1)
    saled = saled * (1 - m5.get_christmas())

    ma28x1 = data.unfold(-2, 28 * 1, 1).mean(-1)
    ma28x1 = torch.nn.functional.pad(ma28x1, (0, 0, 27 + 28 * 1, 0))
    ma28x2 = data.unfold(-2, 28 * 2, 1).mean(-1)
    ma28x2 = torch.nn.functional.pad(ma28x2, (0, 0, 27 + 28 * 2, 0))
    ma28x3 = data.unfold(-2, 28 * 3, 1).mean(-1)
    ma28x3 = torch.nn.functional.pad(ma28x3, (0, 0, 27 + 28 * 3, 0))
    log_ma = torch.cat([ma28x1, ma28x2, ma28x3], -1).clamp(min=1e-3).log()
    del ma28x1, ma28x2, ma28x3  # save memory

    data = data.clamp(min=1e-3).to(args.device)
    covariates = covariates.to(args.device)
    snap = snap.to(args.device)
    dept = dept.to(args.device)
    saled = saled.to(args.device)
    log_ma = log_ma.to(args.device)
    if data.is_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def transform(pred, truth):
        num_samples, duration = pred.size(0), pred.size(-2)
        pred = pred.reshape(num_samples, -1, duration)
        truth = truth.round().reshape(-1, duration).cpu()
        agg_pred = m5.aggregate_samples(pred, *m5.aggregation_levels)
        agg_truth = m5.aggregate_samples(truth.unsqueeze(0), *m5.aggregation_levels).squeeze(0)
        return agg_pred.unsqueeze(-1), agg_truth.unsqueeze(-1)

    def forecaster_options_fn(t0=None, t1=None, t2=None):
        forecaster_options = {
            "create_plates": create_plates,
            "learning_rate": args.learning_rate,
            "learning_rate_decay": args.learning_rate_decay,
            "clip_norm": args.clip_norm,
            "num_steps": args.num_steps,
            "log_every": args.log_every,
            "guide": NormalGuide(create_plates),
        }
        return forecaster_options

    if args.submit:
        pyro.set_rng_seed(args.seed)
        print("Training...")
        forecaster = M5Forecaster(Model(snap, dept, saled, log_ma),
                                  data[:, :, T0:T1],
                                  covariates[T0:T1],
                                  **forecaster_options_fn())

        print("Forecasting...")
        samples = forecaster(data[:, :, T0:T1], covariates[T0:T2], num_samples=1000, batch_size=10)
        samples = samples.reshape(-1, m5.num_timeseries, 28)
        agg_samples = m5.aggregate_samples(samples, *m5.aggregation_levels)
        # cast to numpy because pyro quantile implementation is memory hungry
        print("Calculate quantiles...")
        q = np.quantile(agg_samples.numpy(), m5.quantiles, axis=0)
        print("Make submission...")
        m5.make_uncertainty_submission(args.output_file, q, float_format='%.3f')
    else:
        # calculate weight of each timeseries
        weight = m5.get_aggregated_ma_dollar_sales(m5.aggregation_levels[-1]).cpu()
        weight = weight / weight.sum(0, keepdim=True)
        agg_weight = m5.aggregate_samples(weight.unsqueeze(0), *m5.aggregation_levels).squeeze(0)

        min_train_window = T1 - T0 - args.test_window - (args.num_windows - 1) * args.stride
        print("Backtesting with skip window {}...".format(T0))
        # we will skip crps because it is slow
        metrics = {"mae": eval_mae, "rmse": eval_rmse, "pl": eval_pl}
        windows = m5_backtest(data, covariates[:T1],
                              lambda: Model(snap, dept, saled, log_ma),
                              weight=agg_weight,
                              skip_window=T0,
                              metrics=metrics,
                              transform=transform,
                              forecaster_fn=M5Forecaster,
                              min_train_window=min_train_window,
                              test_window=args.test_window,
                              stride=args.stride,
                              forecaster_options=forecaster_options_fn,
                              num_samples=1000,
                              batch_size=10,
                              seed=args.seed)

        with open(args.output_file, "wb") as f:
            pickle.dump(windows, f)

        for metric in metrics:
            ws_name = "ws_{}".format(metric)
            values = torch.tensor([w[ws_name] for w in windows])
            print("{} = {:0.3g} +- {:0.2g}".format(ws_name, values.mean(), values.std()))


if __name__ == "__main__":
    assert pyro.__version__ >= "1.3.0"
    parser = argparse.ArgumentParser(description="Univariate M5 daily forecasting")
    parser.add_argument("--num-windows", default=3, type=int)
    parser.add_argument("--test-window", default=28, type=int)
    parser.add_argument("-s", "--stride", default=35, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--clip-norm", default=10., type=float)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("-o", "--output-file", default="", type=str)
    parser.add_argument("--submit", action="store_true", default=False)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.output_file == "":
        args.output_file = os.path.basename(__file__)[:-3] + (".csv" if args.submit else ".pkl")
    args.output_file = os.path.join(RESULTS, args.output_file)

    main(args)
