import numpy as np
import torch


def test_zero_calibrator_exactly_preserves_dlinear_prediction():
    from asofcast.calibration import StalenessCalibratedForecaster
    from asofcast.models import BaselineForecaster

    torch.manual_seed(3)
    base = BaselineForecaster(12, 3, 2).eval()
    calibrated = StalenessCalibratedForecaster.from_baseline(base, lookback=12, channels=3, target_channel=2)
    x = torch.randn(5, 12, 3, 4)
    with torch.inference_mode():
        expected = base(x)
        actual = calibrated(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ridge_calibration_reduces_known_causal_residual():
    from asofcast.calibration import StalenessCalibratedForecaster, fit_ridge_calibrator
    from asofcast.models import BaselineForecaster
    from asofcast.training import predict

    rng = np.random.default_rng(12)
    x = rng.normal(size=(320, 12, 3, 4)).astype(np.float32)
    x[..., 1] = (rng.random((320, 12, 3)) > .2).astype(np.float32)
    x[..., 2] = np.abs(x[..., 2])
    x[..., 3] = 1.0
    base = BaselineForecaster(12, 3, 2).eval()
    baseline = predict(base, x)
    causal_residual = .3 * x[:, -1, 0, 0] - .15 * x[:, -1, 1, 2]
    y = (baseline + causal_residual).astype(np.float32)

    model = StalenessCalibratedForecaster.from_baseline(base, lookback=12, channels=3, target_channel=2)
    before = np.mean(np.abs(predict(model, x) - y))
    fit_ridge_calibrator(model, x, y, ridge=1.0)
    after = np.mean(np.abs(predict(model, x) - y))

    assert after < before * .15
    assert model.feature_mean.shape == (13,)
    assert model.correction.weight.numel() == 13


def test_calibrator_features_do_not_use_future_slots():
    from asofcast.calibration import StalenessCalibratedForecaster
    from asofcast.models import BaselineForecaster

    base = BaselineForecaster(6, 2, 1).eval()
    model = StalenessCalibratedForecaster.from_baseline(base, lookback=6, channels=2, target_channel=1)
    x = torch.randn(2, 6, 2, 4)
    with torch.inference_mode():
        features = model.calibration_features(x, base(x))
    assert features.shape == (2, 2 * 4 + 1)
    torch.testing.assert_close(features[:, :2], x[:, -1, :, 0])
    torch.testing.assert_close(features[:, 2:4], x[:, -1, :, 0] - x[:, -2, :, 0])
    torch.testing.assert_close(features[:, 4:6], x[:, -1, :, 2])
    torch.testing.assert_close(features[:, 6:8], x[:, -1, :, 1])
