import re

import duckdb
import numpy as np
import torch
from torch.utils.data import IterableDataset

from msnetloader.utils import (
    CHANNEL_MAP,
    NUM_CHANNELS,
    detect_parquet_schema,
    resolve_active_channels,
    validate_batch_size,
)


class MS2TorchDataset(IterableDataset):

    def __init__(self, parquet_path, batch_size=8, ion_types=("b", "y"), charges=(1, 2),
                 min_consensus_support=None,
                 max_pep=None,
                 extra_where=None
                 ):

        self.min_consensus_support = min_consensus_support
        self.max_pep = max_pep

        conditions = []
        params = []

        if self.min_consensus_support is not None:
            conditions.append("consensus_support >= ?")
            params.append(self.min_consensus_support)

        if self.max_pep is not None:
            conditions.append("posterior_error_probability <= ?")
            params.append(self.max_pep)

        if extra_where is not None:
            conditions.append(f"({extra_where})")

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        schema = detect_parquet_schema(parquet_path)
        if schema == "current":
            columns_sql = """
            sequence,
            peptidoform,
            charge,
            COALESCE(
                list_extract(list_filter(cv_params, x -> x.cv_name = 'instrument'), 1)['cv_value'],
                'unknown'
            ) AS instrument,
            COALESCE(
                CAST(
                    list_extract(
                        list_filter(cv_params, x -> x.cv_name = 'normalized collision energy'), 1
                    )['cv_value']
                    AS DOUBLE
                ),
                0.0
            ) AS nce,
            ion_type_array,
            charge_array,
            intensity_array
            """
        else:
            columns_sql = """
            sequence,
            peptidoform,
            precursor_charge AS charge,
            cv_params.Instrument AS instrument,
            CAST(cv_params."Collision Energy" AS DOUBLE) AS nce,
            ion_type_array,
            charge_array,
            intensity_array
            """

        query = f"""
        SELECT {columns_sql}
        FROM parquet_scan(?)
        {where_clause}
        ORDER BY length(sequence)
        """

        self.query = query
        self.params = [parquet_path] + params
        self.batch_size = validate_batch_size(batch_size)

        self.channel_map = CHANNEL_MAP
        self.active_channels, self.ion_types, self.charges = resolve_active_channels(
            ion_types, charges, self.channel_map
        )

    # -----------------------------
    def __iter__(self):
        # A fresh connection per iteration: the cursor of a previous pass is already
        # consumed, and DataLoader workers must not share a connection across forks.
        con = duckdb.connect()
        reader = con.execute(self.query, self.params).fetch_record_batch(self.batch_size)

        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self.process_batch(batch)

    # -----------------------------
    def process_batch(self, batch):
        sequences = batch["sequence"].to_pylist()
        peptidoform = batch["peptidoform"].to_pylist()
        charges = batch["charge"].to_pylist()
        nces = batch["nce"].to_pylist()
        instruments = batch["instrument"].to_pylist()
        fragments = batch["ion_type_array"].to_pylist()
        fragment_charges = batch["charge_array"].to_pylist()
        intensities = batch["intensity_array"].to_pylist()

        targets = self.build_batch_fragments(
            sequences,
            fragments,
            fragment_charges,
            intensities
        )

        charge_tensor = torch.tensor(charges, dtype=torch.long)
        nce_tensor = torch.tensor(nces, dtype=torch.float32)

        return {
            "peptide": peptidoform,
            "charge": charge_tensor,
            "nce": nce_tensor,
            "instruments": instruments,
            "targets": targets
        }

    # -----------------------------
    def build_batch_fragments(
        self,
        sequences,
        fragments_list,
        frag_charges_list,
        intensity_list
    ):

        B = len(sequences)
        Lmax = max(len(s) for s in sequences)

        out = np.zeros((B, Lmax - 1, NUM_CHANNELS), dtype=np.float32)

        # -----------------------------
        for b in range(B):

            ions = fragments_list[b]
            charges = frag_charges_list[b]
            ints = intensity_list[b]

            if len(ions) == 0:
                continue

            ions = np.asarray(ions)
            charges = np.asarray(charges)
            ints = np.asarray(ints, dtype=np.float32)

            valid = np.array([ion is not None for ion in ions])
            ions = ions[valid]
            charges = charges[valid]
            ints = ints[valid]

            if len(ions) == 0:
                continue

            # remove neutral loss
            mask = np.char.find(ions.astype(str), "-") == -1
            ions = ions[mask]
            charges = charges[mask]
            ints = ints[mask]

            if len(ions) == 0:
                continue

            # -----------------------------
            # parse ion string safely
            # -----------------------------
            ion_str = ions.astype(str)

            ion_type = np.array([x[0] for x in ion_str])

            # SAFE regex position parsing
            pos = np.array([
                int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else -1
                for x in ion_str
            ])

            seq_len = len(sequences[b])

            valid = (pos >= 1) & (pos < seq_len)

            ion_type = ion_type[valid]
            pos = pos[valid] - 1
            charges = charges[valid]
            ints = ints[valid]

            if len(pos) == 0:
                continue

            # -----------------------------
            # channel mapping (fixed)
            # -----------------------------
            ch = np.full(len(ion_type), -1, dtype=np.int32)

            for (t, z), c in self.channel_map.items():
                if t in self.ion_types and z in self.charges:
                    ch[(ion_type == t) & (charges == z)] = c

            valid_ch = ch >= 0

            pos = pos[valid_ch]
            ch = ch[valid_ch]
            ints = ints[valid_ch]

            if len(pos) == 0:
                continue

            # -----------------------------
            # intensity normalize
            # -----------------------------
            max_int = ints.max() if len(ints) > 0 else 1.0
            ints = ints / max_int

            # -----------------------------
            # scatter
            # -----------------------------
            out[b, pos, ch] += ints

        # -----------------------------
        return torch.from_numpy(out[:, :, self.active_channels])
