import numpy as np
import torch


def test_dynamic_quantization_executes_and_preserves_shape():
    from asofcast.models import ArrivalForecaster
    from asofcast.optimize import dynamic_int8
    torch.manual_seed(1)
    model = ArrivalForecaster(12, 3, 2).eval()
    quantized, notes = dynamic_int8(model)
    x = torch.randn(3, 12, 3, 4)
    with torch.inference_mode():
        a, b = model(x), quantized(x)
    assert b.shape == a.shape and torch.isfinite(b).all()
    assert torch.max(torch.abs(a-b)).item() < .1
    assert isinstance(notes, list)


def test_latency_measurement_returns_real_samples():
    from asofcast.models import ArrivalForecaster
    from asofcast.optimize import measure_latency
    result = measure_latency(ArrivalForecaster(12, 3, 2).eval(), torch.ones(1, 12, 3, 4), warmup=2, repeats=10)
    assert result['repeats'] == 10
    assert 0 < result['p50_ms'] <= result['p95_ms']
    assert result['scope'] == 'model_forward_only'
