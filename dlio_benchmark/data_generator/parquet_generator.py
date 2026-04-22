"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Optimized Parquet Generator for DLIO Benchmark.

This implementation uses proper binary/numeric column types instead of
FixedSizeListArray<uint8>, which avoids the 4x file bloat and achieves
6-7x faster write throughput.

Key optimizations:
- Uses pa.large_binary() for binary blob columns (not list<uint8>)
- Uses native scalar types (float32, int8, etc.) for typed columns
- Disables dictionary encoding for random/high-entropy data
- Configurable row group size for optimal read performance
- Memory-efficient batch generation
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

# Map DLIO Compression enum to PyArrow compression strings
# Only include compression types available in the installed Compression enum
COMPRESSION_MAP = {
    Compression.NONE: None,
    Compression.GZIP: 'gzip',
}
# Add optional compression types if available
for comp_name, pq_name in [('SNAPPY', 'snappy'), ('LZ4', 'lz4'), ('ZSTD', 'zstd'), ('BZIP2', 'bz2')]:
    if hasattr(Compression, comp_name):
        COMPRESSION_MAP[getattr(Compression, comp_name)] = pq_name

# NumPy dtype mapping for column generation
NP_DTYPE_MAP = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

# PyArrow scalar type mapping
PA_SCALAR_MAP = {
    'uint8': pa.uint8(),
    'uint16': pa.uint16(),
    'uint32': pa.uint32(),
    'uint64': pa.uint64(),
    'int8': pa.int8(),
    'int16': pa.int16(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'float16': pa.float16(),
    'float32': pa.float32(),
    'float64': pa.float64(),
}


class ParquetGenerator(DataGenerator):
    """
    Optimized Parquet data generator.

    Supports two modes:

    1. **Column-schema mode** (parquet_columns config is non-empty):
       Generates multi-column files from a list of column specifications.
       Each column has a name, dtype, and optional size (for vectors).

    2. **Legacy/binary mode** (parquet_columns is empty):
       Generates single 'data' column with binary blobs, using pa.large_binary()
       instead of the slow FixedSizeListArray<uint8> approach.
    """

    def __init__(self):
        super().__init__()
        # Get parquet-specific configuration
        # Config system loads dataset.parquet.columns -> args.parquet_columns
        # and dataset.parquet.row_group_size -> args.parquet_row_group_size
        self.parquet_columns = getattr(self._args, 'parquet_columns', [])
        self.row_group_size = getattr(self._args, 'parquet_row_group_size', 1024)

    def _build_schema(self, record_size=None):
        """Build PyArrow schema from configuration.

        For column-schema mode, builds schema from parquet_columns config.
        For legacy mode, creates a single 'data' column with large_binary type.
        """
        if not self.parquet_columns:
            # Legacy mode: single binary blob column
            return pa.schema([('data', pa.large_binary())])

        fields = []
        for col_spec in self.parquet_columns:
            name = col_spec.get('name', 'data')
            dtype = col_spec.get('dtype', 'float32')
            size = col_spec.get('size', 1)

            pa_scalar = PA_SCALAR_MAP.get(dtype)

            if pa_scalar is not None:
                if size == 1:
                    # Scalar column
                    fields.append(pa.field(name, pa_scalar))
                else:
                    # Vector column - use large_binary for efficiency
                    # This avoids the FixedSizeListArray performance issue
                    fields.append(pa.field(name, pa.large_binary()))
            elif dtype == 'binary':
                fields.append(pa.field(name, pa.large_binary()))
            elif dtype == 'string':
                fields.append(pa.field(name, pa.string()))
            else:
                # Default to large_binary for unknown types
                fields.append(pa.field(name, pa.large_binary()))

        return pa.schema(fields)

    def _generate_column_data(self, col_spec, num_samples, rng):
        """Generate data for a single column.

        Returns (name, pa.Array) tuple.
        """
        name = col_spec.get('name', 'data')
        dtype = col_spec.get('dtype', 'float32')
        size = col_spec.get('size', 1)

        np_dtype = NP_DTYPE_MAP.get(dtype)
        pa_scalar = PA_SCALAR_MAP.get(dtype)

        if np_dtype is not None and pa_scalar is not None:
            if size == 1:
                # Scalar column - generate flat array
                data = gen_random_tensor(shape=(num_samples,), dtype=np_dtype, rng=rng)
                return name, pa.array(data, type=pa_scalar)
            else:
                # Vector column - generate as binary blobs for efficiency
                # This avoids FixedSizeListArray which has 4x bloat
                # Generate all data at once as 2D array (vectorized, much faster)
                data = gen_random_tensor(shape=(num_samples, size), dtype=np_dtype, rng=rng)
                # Ensure contiguous memory for efficient tobytes()
                if not data.flags['C_CONTIGUOUS']:
                    data = np.ascontiguousarray(data)
                # Convert each row to bytes
                binary_data = [data[i].tobytes() for i in range(num_samples)]
                return name, pa.array(binary_data, type=pa.large_binary())

        if dtype == 'binary':
            binary_data = [rng.bytes(size) for _ in range(num_samples)]
            return name, pa.array(binary_data, type=pa.large_binary())

        if dtype == 'string':
            ints = rng.integers(0, 2**31, size=num_samples)
            strings = [f"s_{v}" for v in ints]
            return name, pa.array(strings, type=pa.string())

        # Fallback: binary blob
        binary_data = [rng.bytes(size) for _ in range(num_samples)]
        return name, pa.array(binary_data, type=pa.large_binary())

    def _generate_legacy_data(self, record_size, num_samples, rng):
        """Generate legacy single-column binary data.

        Uses pa.large_binary() instead of FixedSizeListArray<uint8> for
        6-7x faster writes and correct file sizes.
        """
        binary_data = []
        for _ in range(num_samples):
            data = gen_random_tensor(shape=(record_size,), dtype=np.uint8, rng=rng)
            binary_data.append(data.tobytes())
        return {'data': pa.array(binary_data, type=pa.large_binary())}

    @dlp.log
    def generate(self):
        """Generate Parquet files with optimized encoding."""
        super().generate()

        # Initialize RNG with reproducible seed per rank
        np.random.seed(10 + self.my_rank)
        rng = np.random.default_rng(seed=10 + self.my_rank)

        dim = self.get_dimension(self.total_files_to_generate)
        compression = COMPRESSION_MAP.get(self.compression, None)

        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])

            # Calculate record size from dimensions
            dim_raw = dim[2 * i]
            if isinstance(dim_raw, list):
                dim1 = int(dim_raw[0])
                dim2 = int(dim_raw[1]) if len(dim_raw) > 1 else 1
            else:
                dim1 = int(dim_raw)
                dim2 = int(dim[2 * i + 1])
            record_size = dim1 * dim2

            # Build schema
            schema = self._build_schema(record_size)

            # Generate column data
            if self.parquet_columns:
                columns = {}
                for col_spec in self.parquet_columns:
                    name, array = self._generate_column_data(col_spec, self.num_samples, rng)
                    columns[name] = array
            else:
                columns = self._generate_legacy_data(record_size, self.num_samples, rng)

            # Create table and write
            table = pa.table(columns, schema=schema)

            # Ensure parent directory exists for local filesystem
            parent_dir = os.path.dirname(out_path_spec)
            if parent_dir and not out_path_spec.startswith('s3://'):
                os.makedirs(parent_dir, exist_ok=True)

            # Write with optimizations
            pq.write_table(
                table,
                out_path_spec,
                compression=compression,
                use_dictionary=False,  # Disabled for random/high-entropy data
                row_group_size=self.row_group_size,
                write_statistics=False,  # Skip stats for faster writes
            )

        np.random.seed()
