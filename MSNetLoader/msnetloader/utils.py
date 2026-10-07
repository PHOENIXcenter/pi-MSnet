import numbers
import os

import duckdb

#: Fragment channels produced by the MS2 loaders, in canonical order. Each value is
#: the column index in the target tensor before ``active_channels`` is applied.
CHANNEL_MAP = {
    ("b", 1): 0,
    ("b", 2): 1,
    ("y", 1): 2,
    ("y", 2): 3,
}

#: Width of the unselected target tensor: the highest channel index plus one.
NUM_CHANNELS = max(CHANNEL_MAP.values()) + 1


def detect_parquet_schema(parquet_paths) -> str:
    """Return ``"legacy"`` or ``"current"`` for the MSNet parquet column layout.

    The quantms/msnet collection was re-processed with a new layout
    (``charge``, ``observed_mz``, ``calculated_mz``, ``rt``,
    ``run_file_name`` and a ``(cv_name, cv_value)[]`` ``cv_params`` array);
    files generated before the re-processing use ``precursor_charge``,
    ``exp_mass_to_charge``, ``retention_time``, ``reference_file_name`` and a
    named-struct ``cv_params``. Only the first input file is inspected, so all
    files of one dataset must share a layout.

    Parameters
    ----------
    parquet_paths:
        One or more ``*-MSNet.parquet`` files.

    Returns
    -------
    str
        ``"legacy"`` or ``"current"``.
    """
    if isinstance(parquet_paths, (str, os.PathLike)):
        first = str(parquet_paths)
    else:
        first = str(next(iter(parquet_paths)))
    con = duckdb.connect()
    columns = [
        row[0] for row in con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [first]).fetchall()
    ]
    if "precursor_charge" in columns:
        return "legacy"
    if "charge" in columns:
        return "current"
    raise ValueError(
        f"Unrecognised MSNet parquet layout in {first}: "
        f"expected a 'precursor_charge' or 'charge' column, found {columns}"
    )


def dereduant_precursor(cursor, key="peptidoform"):
    """De-duplicate the rows of an open duckdb cursor on *key*.

    Keeps the first row for every distinct value of *key* and returns the
    remaining rows in their original order.

    Parameters
    ----------
    cursor:
        An open duckdb query result whose columns include *key*.
    key:
        Column to de-duplicate on.

    Returns
    -------
    list[tuple]
        Rows with only the first occurrence kept per distinct *key* value.
    """
    columns = [description[0] for description in cursor.description or []]
    if key not in columns:
        raise ValueError(f"cursor result has no {key!r} column; available columns: {columns}")

    key_index = columns.index(key)
    seen = set()
    output_psms = []
    for row in cursor.fetchall():
        value = row[key_index]
        if value in seen:
            continue
        seen.add(value)
        output_psms.append(row)
    return output_psms


def _as_sequence(value, name):
    """Return *value* as a list, rejecting bare strings and non-iterables."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence such as ('b', 'y'), got the bare string {value!r}"
        )
    try:
        return list(value)
    except TypeError:
        raise TypeError(f"{name} must be an iterable, got {type(value).__name__}") from None


def resolve_active_channels(ion_types=("b", "y"), charges=(1, 2), channel_map=None):
    """Validate *ion_types* and *charges*, and resolve them to target columns.

    Unknown ion types or charges raise :class:`ValueError` instead of being
    silently dropped, which would otherwise yield a zero-width target tensor and
    a model that never learns anything. Duplicate values are collapsed, and the
    returned channels follow *channel_map* order filtered through the caller's
    order, so the channel layout of previously valid calls is unchanged.

    Parameters
    ----------
    ion_types:
        Iterable of fragment ion types, e.g. ``("b", "y")``. A bare string is
        rejected rather than iterated character by character.
    charges:
        Iterable of fragment charge states, e.g. ``(1, 2)``.
    channel_map:
        Mapping of ``(ion_type, charge)`` to target column index. Defaults to
        :data:`CHANNEL_MAP`.

    Returns
    -------
    tuple[list[int], frozenset, frozenset]
        ``(active_channels, ion_types, charges)``, where *active_channels* lists
        the target columns to keep.

    Raises
    ------
    TypeError
        If *ion_types* or *charges* is a bare string or not iterable.
    ValueError
        If a value is unknown, or either argument is empty.
    """
    if channel_map is None:
        channel_map = CHANNEL_MAP
    ion_types = _as_sequence(ion_types, "ion_types")
    charges = _as_sequence(charges, "charges")

    valid_ion_types = {ion_type for ion_type, _ in channel_map}
    valid_charges = {charge for _, charge in channel_map}

    unknown = [
        ion_type for ion_type in ion_types
        if not isinstance(ion_type, str) or ion_type not in valid_ion_types
    ]
    if unknown:
        raise ValueError(
            f"Unknown ion_types {unknown!r}; choose from {sorted(valid_ion_types)!r}"
        )

    # ``not isinstance(..., numbers.Integral)`` short-circuits before the membership
    # test, so unhashable entries cannot raise here. ``bool`` is rejected because
    # ``charges=(True,)`` is far more likely to be a mistake than a request for charge 1.
    unknown = [
        charge for charge in charges
        if isinstance(charge, bool)
        or not isinstance(charge, numbers.Integral)
        or charge not in valid_charges
    ]
    if unknown:
        raise ValueError(f"Unknown charges {unknown!r}; choose from {sorted(valid_charges)!r}")

    # ``dict.fromkeys`` de-duplicates while preserving the order the caller gave, so a
    # repeated ion type can no longer produce the same channel twice in active_channels.
    ion_types = list(dict.fromkeys(ion_types))
    charges = [int(charge) for charge in dict.fromkeys(charges)]

    if not ion_types:
        raise ValueError("ion_types must contain at least one value")
    if not charges:
        raise ValueError("charges must contain at least one value")

    active_channels = [
        channel_map[(ion_type, charge)]
        for ion_type in ion_types
        for charge in charges
        if (ion_type, charge) in channel_map
    ]
    if not active_channels:
        raise ValueError(
            f"No channel matches ion_types={ion_types!r} and charges={charges!r}; "
            f"choose from {sorted(channel_map)!r}"
        )
    return active_channels, frozenset(ion_types), frozenset(charges)


def validate_batch_size(batch_size, name="batch_size") -> int:
    """Validate *batch_size* and return it as a plain ``int``.

    Parameters
    ----------
    batch_size:
        Number of rows per arrow record batch; must be an integer >= 1.
    name:
        Parameter name used in the error message.

    Returns
    -------
    int
        *batch_size* as a builtin ``int``.

    Raises
    ------
    ValueError
        If *batch_size* is not integral or is less than 1, matching the
        ``ValueError`` the rest of the package raises for bad arguments.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= 1, got {batch_size!r}")
    if batch_size < 1:
        raise ValueError(f"{name} must be >= 1, got {batch_size!r}")
    return int(batch_size)
