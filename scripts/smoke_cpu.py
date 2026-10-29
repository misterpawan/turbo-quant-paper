"""Offline CPU checks of the codec, padding, reconstruction, and paired CI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
import numpy as np
import torch
from TurboLDM_experiments_helper import (
    fwht, seed_everything, turboquant_vector_quantize, uniform_group_quantize,
    UniformLinear, TurboLDMLinear, StageContext,
)
from bootstrap_ci import paired_bootstrap


def main():
    seed_everything(42)
    x = torch.randn(4, 128)
    torch.testing.assert_close(fwht(fwht(x)), x, atol=1e-5, rtol=1e-5)
    layer = torch.nn.Linear(137, 16)
    inputs = torch.randn(3, 137)
    for bits in (3, 4):
        for codec, cls in ((uniform_group_quantize, UniformLinear),
                           (turboquant_vector_quantize, TurboLDMLinear)):
            seed_everything(42)
            pack = codec(layer.weight.detach(), bits=bits)
            seed_everything(42)
            repeat = codec(layer.weight.detach(), bits=bits)
            torch.testing.assert_close(pack['deq'], repeat['deq'])
            assert pack['deq'].shape == layer.weight.shape
            assert torch.isfinite(pack['deq']).all()
            mod = cls(layer, pack, None, ()) if cls is TurboLDMLinear else cls(layer, pack)
            actual = mod(inputs)
            expected = torch.nn.functional.linear(inputs, pack['deq'], layer.bias)
            torch.testing.assert_close(actual, expected)
            assert mod.quant_storage_bytes() > 0
            print(f'{codec.__name__}: {bits}-bit padded forward OK')
    base = turboquant_vector_quantize(layer.weight.detach(), bits=3)
    residual = turboquant_vector_quantize(layer.weight.detach() - base['deq'], bits=1)
    mod = TurboLDMLinear(layer, base, residual, (1,))
    previous_stage = StageContext.current_stage
    try:
        for stage in (0, 1, 2):
            StageContext.current_stage = stage
            weight = base['deq'] + residual['deq'] if stage == 1 else base['deq']
            torch.testing.assert_close(mod(inputs), torch.nn.functional.linear(inputs, weight, layer.bias))
    finally:
        StageContext.current_stage = previous_stage
    print('stage-aware residual: active and inactive stages OK')
    a = np.array([1., 0., 1., 0.])
    assert paired_bootstrap(a, a) == (0., 0.)
    print('PASS: FWHT inverse, seeded codecs, padding, linear reconstruction, paired CI')


if __name__ == '__main__':
    main()
