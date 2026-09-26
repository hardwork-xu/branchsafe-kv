"""Offline model-layout adapter tests. / 离线模型缓存布局适配测试。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_model.py"
SPEC = importlib.util.spec_from_file_location("validate_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


@pytest.mark.parametrize("tokens", [0, 1, 15, 16, 17, 65])
def test_native_layout_roundtrip_and_independence(tokens: int) -> None:
    random = np.random.default_rng(41)
    layers = tuple(
        (
            random.standard_normal((1, 3, tokens, 7), dtype=np.float32),
            random.standard_normal((1, 3, tokens, 7), dtype=np.float32),
        )
        for _ in range(2)
    )
    k, v = ADAPTER.pack_layers(layers)
    assert k.shape == (2, tokens, 3, 7)
    restored = ADAPTER.unpack_layers(k, v)
    for expected_pair, actual_pair in zip(layers, restored, strict=True):
        for expected, actual in zip(expected_pair, actual_pair, strict=True):
            np.testing.assert_array_equal(actual, expected)
            assert not np.shares_memory(actual, expected)
    if tokens:
        assert k[1, 0, 2, 3] == layers[1][0][0, 2, 0, 3]
        k.fill(0)
        np.testing.assert_array_equal(restored[0][0], layers[0][0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(bad: float) -> None:
    array = np.zeros((1, 2, 3, 4), dtype=np.float32)
    array[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ADAPTER.pack_layers([(array, array)])
    with pytest.raises(ValueError, match="finite"):
        ADAPTER.unpack_layers(array, array)


def test_invalid_layout_rejected() -> None:
    with pytest.raises(ValueError, match="one layer"):
        ADAPTER.pack_layers([])
    good = np.zeros((1, 2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="batch=1"):
        ADAPTER.pack_layers([(np.zeros((2, 2, 3, 4), dtype=np.float32), good)])
    with pytest.raises(ValueError, match="dtype"):
        ADAPTER.pack_layers([(good.astype(np.float64), good)])
    with pytest.raises(ValueError, match="four-dimensional"):
        ADAPTER.unpack_layers(good, good[:, :, :2])
    with pytest.raises(ValueError, match="float32"):
        ADAPTER.unpack_layers(good.astype(np.float64), good.astype(np.float64))
