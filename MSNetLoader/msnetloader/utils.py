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
