import numpy as np
import pytest
import torch


def test_dlinear_shape_constant_and_backward():
    from asofcast.models import DLinear
    model = DLinear(12, 3, 6, kernel_size=5)
    x = torch.full((4, 12, 3), 2.0)
    y = model(x)
    assert y.shape == (4, 6, 3)
    torch.testing.assert_close(y, torch.full_like(y, 2.0))
    y.sum().backward()
    assert model.trend.weight.grad is not None


def test_decomposition_reconstructs_signal():
    from asofcast.models import decompose
    x = torch.randn(4, 12, 3)
    seasonal, trend = decompose(x, 5)
    torch.testing.assert_close(seasonal + trend, x)
    assert trend.shape == x.shape


@pytest.mark.parametrize('kernel', [0, 2, -1])
def test_invalid_decomposition_kernel(kernel):
    from asofcast.models import DLinear
    with pytest.raises(ValueError): DLinear(12, 3, 1, kernel_size=kernel)


def test_arrival_model_uses_value_and_metadata_channels():
    from asofcast.models import ArrivalForecaster
    torch.manual_seed(7)
    model = ArrivalForecaster(12, 3, 2)
    x = torch.randn(4, 12, 3, 4, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)
    out.sum().backward()
    assert x.grad[..., 0].abs().sum() > 0
    assert x.grad[..., 1:].abs().sum() > 0


def test_actual_training_reduces_toy_error_and_changes_weights():
    from asofcast.models import BaselineForecaster
    from asofcast.training import predict, train_regressor
    rng = np.random.default_rng(8)
    x = rng.normal(size=(128, 12, 3, 4)).astype(np.float32)
    y = (x[:, -1, 2, 0] * .8 + .2).astype(np.float32)
    torch.manual_seed(8)
    model = BaselineForecaster(12, 3, 2)
    before = np.mean((predict(model, x) - y) ** 2)
    original = model.dlinear.seasonal.weight.detach().clone()
    history = train_regressor(model, x, y, epochs=40, learning_rate=.02, batch_size=64, seed=8)
    after = np.mean((predict(model, x) - y) ** 2)
    assert after < before * .1
    assert not torch.equal(original, model.dlinear.seasonal.weight)
    assert len(history) == 40 and np.isfinite(history).all()


def test_invalid_training_data_rejected():
    from asofcast.models import BaselineForecaster
    from asofcast.training import train_regressor
    model = BaselineForecaster(12, 3, 2)
    with pytest.raises(ValueError):
        train_regressor(model, np.zeros((3, 12, 3, 4)), np.ones(2), epochs=1)
