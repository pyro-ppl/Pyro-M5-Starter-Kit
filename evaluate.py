# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.contrib.forecast.evaluate import backtest, eval_crps, eval_mae, logger
from pyro.ops.stats import quantile

from util import M5Data


@torch.no_grad()
def eval_rmse(pred, truth):
    """
    Like pyro.contrib.forecast.eval_rmae but does not average over batch dimensions.
    """
    pred = pred.mean(0)
    error = pred - truth
    return (error * error).reshape(truth.shape[:-2] + (-1,)).mean(-1).sqrt()


@torch.no_grad()
def eval_pl(pred, truth):
    """
    Computes pinball loss over 9 quantiles 0.005, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995.
    """
    us = torch.tensor(M5Data.quantiles, dtype=pred.dtype, device=pred.device)
    pred = quantile(pred, probs=us, dim=0)  # 9 x batch_shape x duration x D
    error = pred - truth.unsqueeze(0)
    us = us.reshape((-1,) + (1,) * (pred.dim() - 1)).expand(pred.shape)
    error = (torch.where(error <= 0, us, 1 - us) * error).mean(0)  # mean accross all quantiles
    return error.reshape(truth.shape[:-2] + (-1,)).mean(-1)


DEFAULT_METRICS = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "pl": eval_pl,
}


def eval_wrmsse(raw_value, train_data, weight):
    lag1 = train_data[..., 1:, :] - train_data[..., :-1, :]
    # drop starting 0s
    actual_length = (train_data.sum(-1).cumsum(-1) != 0).sum(-1)
    scale = lag1.pow(2).mean(-1).sum(-1).div(actual_length - 1).sqrt()
    return (weight * raw_value / scale).sum().cpu().item()


def eval_wspl(raw_value, train_data, weight):
    lag1 = train_data[..., 1:, :] - train_data[..., :-1, :]
    # drop starting 0s
    actual_length = (train_data.sum(-1).cumsum(-1) != 0).sum(-1)
    scale = lag1.abs().mean(-1).sum(-1).div(actual_length - 1)
    return (weight * raw_value / scale).sum().cpu().item()


def m5_backtest(data, covariates, model_fn, weight=None, **kwargs):
    """
    Backtest function with weighted metrics. See
    http://docs.pyro.ai/en/stable/contrib.forecast.html#pyro.contrib.forecast.evaluate.backtest
    for more information.

    .. note:: In M5 competition, joint result for all aggregation levels is
        the average of the results at each aggregation level.

    :param torch.Tensor weight: weight of each time series in `data`.
        This should satisfy `weight.shape == data.shape[:-1]`.
    """
    if weight is None:
        weight = data.new_ones(data.shape[:-1])
    assert weight.shape == data.shape[:-1]
    # normalize over batch dimensions
    weight = weight / weight.reshape((-1, data.shape[-2])).sum(0)

    if kwargs.get("metrics") is None:
        kwargs["metrics"] = DEFAULT_METRICS

    min_train_window = (data.sum(-1).cumsum(-1) == 0).sum(-1).max().cpu().item() + 2
    if kwargs.get("min_train_window", 1) < min_train_window:
        logger.info(f"min_train_window is set to {min_train_window} "
                    "to be able to compute scaled metrics.")
        kwargs["min_train_window"] = min_train_window

    windows = backtest(data, covariates, model_fn, **kwargs)
    for window in windows:
        # we use all historical data before t1 to compute
        # the scale factor of wrmsse and wspl
        train_data = data[..., :window["t1"], :]
        weight = weight[window["t1"] - 1]

        if "rmse" in window:
            window["wrmsse"] = eval_wrmsse(window["rmse"], train_data, weight)

        if "pl" in window:
            window["wspl"] = eval_wspl(window["pl"], train_data, weight)

    return windows
