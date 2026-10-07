"""Unit tests for the argument validation helpers in ``msnetloader.utils``.

The MS2 loaders used to derive their channels from an unvalidated
``set(ion_types)`` / ``set(charges)`` pair and drop anything the channel map did
not know about, so an unknown value silently produced a zero-width target
tensor and duplicate values produced duplicate channels.
"""

import numpy as np
import pytest

from msnetloader.utils import (
    CHANNEL_MAP,
    NUM_CHANNELS,
    resolve_active_channels,
    validate_batch_size,
)


def test_num_channels_covers_every_channel_index():
    assert NUM_CHANNELS == max(CHANNEL_MAP.values()) + 1


def test_resolve_active_channels_default():
    assert resolve_active_channels() == ([0, 1, 2, 3], frozenset({"b", "y"}), frozenset({1, 2}))


def test_resolve_active_channels_dedupes_repeated_values():
    channels, ion_types, charges = resolve_active_channels(("b", "b", "y", "y"), (1, 1, 2, 2))

    assert channels == [0, 1, 2, 3]
    assert ion_types == frozenset({"b", "y"})
    assert charges == frozenset({1, 2})


def test_resolve_active_channels_preserves_caller_order():
    # ("y", "b") keeps the y channels first, exactly as it did before validation existed.
    assert resolve_active_channels(("y", "b"), (1, 2))[0] == [2, 3, 0, 1]


def test_resolve_active_channels_accepts_a_subset():
    assert resolve_active_channels(("b",), (1,))[0] == [0]


def test_resolve_active_channels_accepts_numpy_input():
    channels, ion_types, charges = resolve_active_channels(np.array(["b", "y"]), np.array([1, 2]))

    assert channels == [0, 1, 2, 3]
    assert ion_types == frozenset({"b", "y"})
    assert charges == frozenset({1, 2})


@pytest.mark.parametrize("ion_types", [("z",), ("b", "z"), ("B", "Y")])
def test_resolve_active_channels_rejects_unknown_ion_type(ion_types):
    with pytest.raises(ValueError, match="Unknown ion_types"):
        resolve_active_channels(ion_types, (1, 2))


@pytest.mark.parametrize("charges", [(3,), (0,), (-1,), (1, 3)])
def test_resolve_active_channels_rejects_unknown_charge(charges):
    with pytest.raises(ValueError, match="Unknown charges"):
        resolve_active_channels(("b", "y"), charges)


@pytest.mark.parametrize("charges", [(True,), (1.5,), ("1",)])
def test_resolve_active_channels_rejects_non_integer_charge(charges):
    with pytest.raises(ValueError, match="Unknown charges"):
        resolve_active_channels(("b", "y"), charges)


def test_resolve_active_channels_rejects_empty_ion_types():
    with pytest.raises(ValueError, match="ion_types must contain at least one value"):
        resolve_active_channels((), (1, 2))


def test_resolve_active_channels_rejects_empty_charges():
    with pytest.raises(ValueError, match="charges must contain at least one value"):
        resolve_active_channels(("b", "y"), ())


@pytest.mark.parametrize("value", ["by", b"by"])
def test_resolve_active_channels_rejects_a_bare_string(value):
    with pytest.raises(TypeError, match="must be a sequence"):
        resolve_active_channels(value, (1, 2))


@pytest.mark.parametrize("value", [5, None])
def test_resolve_active_channels_rejects_non_iterable(value):
    with pytest.raises(TypeError, match="must be an iterable"):
        resolve_active_channels(("b", "y"), value)


@pytest.mark.parametrize("batch_size", [0, -1, -100])
def test_validate_batch_size_rejects_non_positive(batch_size):
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_batch_size(batch_size)


@pytest.mark.parametrize("batch_size", [2.5, "8", True, None])
def test_validate_batch_size_rejects_non_integer(batch_size):
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        validate_batch_size(batch_size)


def test_validate_batch_size_accepts_and_normalises_numpy_int():
    result = validate_batch_size(np.int64(8))

    assert result == 8
    assert isinstance(result, int)


def test_validate_batch_size_uses_the_given_name():
    with pytest.raises(ValueError, match="rows_per_batch"):
        validate_batch_size(-1, name="rows_per_batch")
