# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from pyro.contrib.forecast.evaluate import backtest, logger
from pyro.ops.stats import crps_empirical

from util import M5Data


@torch.no_grad()
def eval_mae(pred, truth):
    """
    Like pyro.contrib.forecast.eval_mae but does not average over batch dimensions.
    """
    logger.info("Evaluating MAE...")
    pred = pred.median(0).values
    return (pred - truth).abs().reshape(truth.shape[:-2] + (-1,)).mean(-1)


@torch.no_grad()
def eval_rmse(pred, truth):
    """
    Like pyro.contrib.forecast.eval_rmse but does not average over batch dimensions.
    """
    logger.info("Evaluating RMSE...")
    pred = pred.mean(0)
    error = pred - truth
    return (error * error).reshape(truth.shape[:-2] + (-1,)).mean(-1).sqrt()


@torch.no_grad()
def eval_crps(pred, truth):
    """
    Like pyro.contrib.forecast.eval_crps but does not average over batch dimensions.
    """
    logger.info("Evaluating CRPS...")
    return crps_empirical(pred, truth).reshape(truth.shape[:-2] + (-1,)).mean(-1)


@torch.no_grad()
def eval_pl(pred, truth):
    """
    Computes pinball loss over 9 quantiles 0.005, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995.
    """
    logger.info("Evaluating PL...")
    us = torch.tensor(M5Data.quantiles, dtype=pred.dtype, device=pred.device)
    # pred = quantile(pred, probs=us, dim=0)  # 9 x batch_shape x duration x D
    # TODO: improve the speed of pyro.ops.stats.quantile to use it here
    pred = torch.from_numpy(np.quantile(pred.cpu().numpy(), M5Data.quantiles, axis=0)).to(
        pred.device).type(pred.dtype)
    error = pred - truth.unsqueeze(0)
    us = us.reshape((-1,) + (1,) * (pred.dim() - 1)).expand(pred.shape)
    error = torch.where(error <= 0, -us, 1 - us).mul(error).mean(0)  # mean accross all quantiles
    return error.reshape(truth.shape[:-2] + (-1,)).mean(-1)


DEFAULT_METRICS = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "pl": eval_pl,
}


def get_metric_scale(metric, train_data):
    duration = train_data.shape[-2]
    lag1 = train_data - torch.nn.functional.pad(train_data[..., :-1, :], (0, 0, 1, 0))
    # find active time: to drop the leading 0s
    active_time = (train_data.sum(-1, keepdims=True).cumsum(-2) != 0).sum(-2, keepdims=True)
    # resolve the edge case: the item is not active during the backtesting train window
    # there are two situations here:
    #   + the starting day happens after the last day of train_data: we let scale = 1
    #   + the starting day is the last day: we let scale get the value of the last day
    active_time = active_time.clamp(min=2)
    start_value = train_data.gather(
        -2, (duration - active_time).expand(active_time.shape[:-1] + train_data.shape[-1:]))
    norm = 2 if metric == "rmse" else 1
    lag1_norm = lag1.abs().pow(norm).sum(-2) - start_value.squeeze(-2).abs().pow(norm)
    # return 1. if train_data is all zeros, this does not matter because the weight is 0.
    lag1_norm = lag1_norm.clamp(min=1.)
    return lag1_norm.mean(-1).div(active_time.squeeze(-1).squeeze(-1) - 1).pow(1 / norm)


@torch.no_grad()
def eval_weighted_scale(metric, value, train_data, weight):
    scale = get_metric_scale(metric, train_data)
    return (weight * value / scale).sum().cpu().item()


def m5_backtest(data, covariates, model_fn, weight=None, skip_window=0, **kwargs):
    """
    Backtest function with weighted metrics. See
    http://docs.pyro.ai/en/stable/contrib.forecast.html#pyro.contrib.forecast.evaluate.backtest
    for more information.

    If `data` is transformed from a raw timeseries, make sure that a keyword `transform` is
    provided to transform the prediction back to the original scale.

    .. note:: In M5 competition, joint result for all aggregation levels is
        the average of the results at each aggregation level.

    :param torch.Tensor weight: weight of each time series in the raw data (after transform).
    :param int skip_window: skip a small leading period of data and covariates
    """
    if kwargs.get("metrics") is None:
        kwargs["metrics"] = DEFAULT_METRICS

    transform = kwargs.get("transform")
    raw_data = data if transform is None else transform(data.unsqueeze(0), data)[1]
    if weight is None:
        weight = raw_data.new_ones(raw_data.shape[:-1])
    assert weight.shape == raw_data.shape[:-1]
    # normalize over batch dimensions
    weight_norm = weight.reshape((-1, raw_data.shape[-2])).sum(0)
    weight = weight / weight_norm

    windows = backtest(data[..., skip_window:, :], covariates[..., skip_window:, :],
                       model_fn, **kwargs)
    for window in windows:
        window["t0"] += skip_window
        window["t1"] += skip_window
        window["t2"] += skip_window
        # we use all historical data before t1 to compute
        # the scale factor of wrmsse and wspl
        train_data = raw_data[..., :window["t1"], :]
        w = weight[..., window["t1"] - 1]

        for metric in kwargs["metrics"].keys():
            window[f"ws_{metric}"] = eval_weighted_scale(metric, window[metric], train_data, w)

    return windows
