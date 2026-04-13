import tensorflow as tf
import numpy as np
import duckdb
import re


class MS2TFDataset:

    def __init__(
        self,
        parquet_path,
        batch_size=8,
        ion_types=("b", "y"),
        charges=(1, 2),
        min_consensus_support=None,
        max_pep=None
    ):
        con = duckdb.connect()
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

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT
            sequence,
            peptidoform,
            precursor_charge AS charge,
            cv_params.Instrument AS instrument,
            CAST(cv_params."Collision Energy" AS DOUBLE) AS nce,
            ion_type_array,
            charge_array,
            intensity_array
        FROM parquet_scan(?)
        {where_clause}
        ORDER BY length(sequence)
        """

        self.cursor = con.execute(query, [parquet_path] + params)

        self.batch_size = batch_size

        self.ion_types = set(ion_types)
        self.charges = set(charges)

        self.channel_map = {
            ("b", 1): 0,
            ("b", 2): 1,
            ("y", 1): 2,
            ("y", 2): 3,
        }

        self.active_channels = [
            self.channel_map[(t, z)]
            for t in ion_types for z in charges
            if (t, z) in self.channel_map
        ]

    # =========================================================
    # generator（核心替代 __iter__）
    # =========================================================
    def generator(self):
        reader = self.cursor.fetch_record_batch(self.batch_size)
        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self.process_batch(batch)

    # =========================================================
    def get_dataset(self):
        output_signature = {
            "peptide": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "charge": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "nce": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "instruments": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "targets": tf.TensorSpec(shape=(None, None, len(self.active_channels)), dtype=tf.float32),
        }

        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

    # =========================================================
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

        return {
            "peptide": np.array(peptidoform, dtype=np.string_),
            "charge": np.array(charges, dtype=np.int32),
            "nce": np.array(nces, dtype=np.float32),
            "instruments": np.array(instruments, dtype=np.string_),
            "targets": targets.numpy(),  # TF expects numpy
        }

    # =========================================================
    def build_batch_fragments(
        self,
        sequences,
        fragments_list,
        frag_charges_list,
        intensity_list
    ):
        B = len(sequences)
        Lmax = max(len(s) for s in sequences)

        out = np.zeros((B, Lmax - 1, 4), dtype=np.float32)

        for b in range(B):
            ions = fragments_list[b]
            charges = frag_charges_list[b]
            ints = intensity_list[b]

            if len(ions) == 0:
                continue

            ions = np.asarray(ions)
            charges = np.asarray(charges)
            ints = np.asarray(ints, dtype=np.float32)

            valid = (ions != None)
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

            ion_str = ions.astype(str)
            ion_type = np.array([x[0] for x in ion_str])

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

            max_int = ints.max() if len(ints) > 0 else 1.0
            ints = ints / max_int

            out[b, pos, ch] += ints

        return tf.convert_to_tensor(out[:, :, self.active_channels], dtype=tf.float32)