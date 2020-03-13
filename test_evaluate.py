# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from evaluate import DEFAULT_METRICS, eval_weighted_scale


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
@pytest.mark.parametrize("obs_dim", [1, 5])
@pytest.mark.parametrize("metric", list(DEFAULT_METRICS.keys()))
def test_weighted_scale_metric(batch_shape, obs_dim, metric):
    duration = 30
    active_time = 5
    forecast_time = 20
    data = torch.randint(0, 10, batch_shape + (duration, obs_dim)).float()
    data[..., :active_time, :] = 0.
    data[..., active_time, :] += 1.
    weight = torch.randn(batch_shape).exp()
    train_data, truth = data[..., :forecast_time, :], data[..., forecast_time:, :]
    active_data = train_data[..., active_time:, :]

    num_samples = 7
    pred = torch.randn((num_samples,) + truth.shape).exp()
    value = DEFAULT_METRICS[metric](pred, truth)
    assert value.shape == batch_shape
    assert (value >= 0).all()
    actual = eval_weighted_scale(metric, value, train_data, weight)
    expected = eval_weighted_scale(metric, value, active_data, weight)
    assert actual == expected
