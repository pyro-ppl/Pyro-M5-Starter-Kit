# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.contrib.forecast import backtest as pyro_backtest
from pyro.ops.stats import crps_empirical, quantile

from util import M5Data


@torch.no_grad()
def eval_mae(pred, truth):
    """
    Like pyro.contrib.forecast.eval_mae but does not average over batch dimensions.
    """
    pred = pred.median(0).values
    return (pred - truth).abs().reshape(truth.shape[:-2] + (-1,)).mean(-1).cpu()


@torch.no_grad()
def eval_rmse(pred, truth):
    """
    Like pyro.contrib.forecast.eval_rmae but does not average over batch dimensions.
    """
    pred = pred.mean(0)
    error = pred - truth
    return (error * error).reshape(truth.shape[:-2] + (-1,)).mean(-1).sqrt().cpu()


@torch.no_grad()
def eval_crps(pred, truth):
    """
    Like pyro.contrib.forecast.eval_crps but does not average over batch dimensions.
    """
    return crps_empirical(pred, truth).reshape(truth.shape[:-2] + (-1,)).mean(-1).cpu()


@torch.no_grad()
def eval_pl(pred, truth):
    """
    Computes pinpall loss over 9 quantiles 0.005, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995.
    """
    us = torch.tensor(M5Data.quantiles, dtype=pred.dtype, device=pred.device)
    pred = quantile(pred, probs=us, dim=0)  # 9 x batch_shape x duration x D
    error = pred - truth.unsqueeze(0)
    us = us.reshape((-1,) + (1,) * (pred.dim() - 1)).expand(pred.shape)
    error = (torch.where(error <= 0, us, 1 - us) * error).mean(0)  # mean accross all quantiles
    return error.reshape(truth.shape[:-2] + (-1,)).mean(-1).cpu()


DEFAULT_METRICS = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "pl": eval_pl,
}


def get_pl_scale(data):
    lag1 = data[..., 1:, :] - data[..., :-1, :]
    # drop starting 0s
    actual_length = (data.cumsum(-2) != 0).sum(-2) - 1
    return lag1.abs().sum(-2).div(actual_length)


def get_rmse_scale(data):
    lag1 = data[..., 1:, :] - data[..., :-1, :]
    # drop starting 0s
    actual_length = (data.cumsum(-2) != 0).sum(-2) - 1
    return lag1.pow(2).sum(-2).div(actual_length).sqrt()


def backtest(data, covariates, model_fn, weight=torch.tensor(1.), **kwargs):
    """
    Backtest function with weighted metrics. See
    http://docs.pyro.ai/en/stable/contrib.forecast.html#pyro.contrib.forecast.evaluate.backtest
    for more information.

    .. note:: In M5 competition, joint result for all aggregation levels is
        the average of the results at each aggregation level.

    :param torch.Tensor weight: weight of each time series in `data`.
        This should satisfy `weight.shape == data.shape[:-2]`.
    """
    assert weight.shape == data.shape[:-2]
    weight = weight / weight.sum()

    if kwargs.get("metrics") is None:
        kwargs["metrics"] = DEFAULT_METRICS

    windows = pyro_backtest(data, covariates, model_fn, **kwargs)
    for window in windows:
        if "rmse" in window:
            train_data = data[..., window["t0"]:window["t1"], :]
            scale = get_rmse_scale(train_data)
            rmsse = window["rmse"] / scale
            assert rmsse.shape == weight.shape
            window["wrmsse"] = (rmsse * weight).sum().item()

        if "pl" in window:
            train_data = data[..., window["t0"]:window["t1"], :]
            scale = get_pl_scale(train_data)
            spl = window["pl"] / scale
            assert spl.shape == weight.shape
            window["wspl"] = (rmsse * weight).sum().item()

    return windows
