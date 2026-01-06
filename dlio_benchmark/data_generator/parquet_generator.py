
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
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)


class ParquetGenerator(DataGenerator):
    """
    Generator for creating data in Parquet columnar format.
    
    Parameters:
        field_specs: Dictionary mapping field names to their specifications.
                    Each spec is a dict with:
                    - 'dtype': numpy dtype or python type
                    - 'shape': tuple for fixed-size arrays, or None for scalars
                    - 'variable_length': bool, whether this field has variable length
                    - 'length': fixed length for non-variable fields (optional)
                    - 'avg_length': average length for variable-length fields
                    - 'std_length': standard deviation of length for variable-length fields
                    
    Example field_specs:
        {
            'image_data': {
                'dtype': np.float32,
                'shape': (224, 224, 3),
                'variable_length': False
            },
            'label': {
                'dtype': np.int64,
                'shape': None,
                'variable_length': False
            },
            'fixed_metadata': {
                'dtype': str,
                'shape': None,
                'variable_length': False,
                'length': 100  # Always 100 characters
            },
            'variable_metadata': {
                'dtype': str,
                'shape': None,
                'variable_length': True,
                'avg_length': 100,
                'std_length': 20
            },
            'fixed_embedding': {
                'dtype': np.float32,
                'shape': (128,),  # 128-dimensional embedding
                'variable_length': False,
                'length': 512  # Override to 512 dimensions
            },
            'variable_embedding': {
                'dtype': np.float32,
                'shape': None,
                'variable_length': True,
                'avg_length': 512,
                'std_length': 50
            }
        }
    """
    
    @dlp.log_init
    def __init__(self):
        super().__init__()
        
        # Get field specifications
        self.field_specs = getattr(self._args, 'parquet_field_specs', None)
        if self.field_specs is None:
            self.field_specs = self._get_default_field_specs()
        
        # Compression settings
        self.parquet_compression = getattr(self._args, 'compression', None)
        self.compression_level = getattr(self._args, 'compression_level', None)
        
        # Row group settings - NEW TUNABLE PARAMETERS
        self.row_group_size = getattr(self._args, 'parquet_row_group_size', None)
        # If not specified, PyArrow will use default (typically 1M rows or 64MB)
        
        # Alternative: specify number of row groups per file
        self.num_row_groups_per_file = getattr(self._args, 'parquet_num_row_groups', None)

    def _get_default_field_specs(self):
        """
        Provide default field specifications based on record dimensions.
        """
        dim = self.get_dimension(1)[0]
        
        return {
            'data': {
                'dtype': self._args.record_element_dtype,
                'shape': tuple(dim) if isinstance(dim, list) else (dim,),
                'variable_length': False
            },
            'label': {
                'dtype': np.int64,
                'shape': None,
                'variable_length': False
            }
        }

    def _generate_field_data(self, field_name, field_spec, num_samples, rng):
        """Generate data for a single field according to its specification."""
        dtype = field_spec.get('dtype', np.float32)
        shape = field_spec.get('shape', None)
        variable_length = field_spec.get('variable_length', False)
        
        if variable_length:
            avg_length = field_spec.get('avg_length', 100)
            std_length = field_spec.get('std_length', 20)
            
            if dtype == str or dtype == 'str':
                # Variable-length strings
                data = []
                for _ in range(num_samples):
                    length = max(1, int(rng.normal(avg_length, std_length)))
                    data.append(''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), length)))
                return data
            elif dtype == bytes or dtype == 'bytes':
                # Variable-length bytes
                data = []
                for _ in range(num_samples):
                    length = max(1, int(rng.normal(avg_length, std_length)))
                    data.append(rng.bytes(length))
                return data
            else:
                # Variable-length numeric arrays
                data = []
                for _ in range(num_samples):
                    length = max(1, int(rng.normal(avg_length, std_length)))
                    if shape:
                        arr_shape = (length,) + shape
                    else:
                        arr_shape = (length,)
                    data.append(gen_random_tensor(arr_shape, dtype, rng))
                return data
        else:
            # Fixed-length data
            length = field_spec.get('length', None)
            
            if dtype == str or dtype == 'str':
                # Fixed-length strings
                if length is None:
                    length = 100
                data = [''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), length)) 
                        for _ in range(num_samples)]
                return data
            elif dtype == bytes or dtype == 'bytes':
                # Fixed-length bytes
                if length is None:
                    length = 100
                data = [rng.bytes(length) for _ in range(num_samples)]
                return data
            else:
                # Fixed-length numeric arrays
                if shape:
                    if length is not None:
                        # Override the first dimension with specified length
                        data_shape = (num_samples, length) + shape[1:] if len(shape) > 1 else (num_samples, length)
                    else:
                        data_shape = (num_samples,) + shape
                else:
                    # Scalar values
                    data_shape = (num_samples,)
                
                data = gen_random_tensor(data_shape, dtype, rng)
                return data

    def _numpy_to_arrow_type(self, dtype):
        """
        Convert numpy dtype to PyArrow type.
        
        Args:
            dtype: numpy dtype or python type
            
        Returns:
            PyArrow type
        """
        if dtype == str or dtype == 'str':
            return pa.string()
        elif dtype == bytes or dtype == 'bytes':
            return pa.binary()
        elif dtype == np.float32:
            return pa.float32()
        elif dtype == np.float64:
            return pa.float64()
        elif dtype == np.int32:
            return pa.int32()
        elif dtype == np.int64:
            return pa.int64()
        elif dtype == np.uint8:
            return pa.uint8()
        elif dtype == np.uint16:
            return pa.uint16()
        elif dtype == np.uint32:
            return pa.uint32()
        elif dtype == np.uint64:
            return pa.uint64()
        else:
            # For other types, try to infer from numpy
            return pa.from_numpy_dtype(dtype)

    def _create_arrow_schema(self):
        """
        Create PyArrow schema from field specifications.
        
        Returns:
            PyArrow schema
        """
        fields = []
        
        for field_name, field_spec in self.field_specs.items():
            dtype = field_spec['dtype']
            shape = field_spec.get('shape', None)
            variable_length = field_spec.get('variable_length', False)
            
            # Get base Arrow type
            base_type = self._numpy_to_arrow_type(dtype)
            
            # Special handling for string and bytes types
            if dtype == str or dtype == 'str' or dtype == bytes or dtype == 'bytes':
                # String and bytes types are inherently variable-length in Arrow
                # Just use the base type directly
                arrow_type = base_type
            elif variable_length:
                # Variable-length fields use list type for numeric arrays
                arrow_type = pa.list_(base_type)
            else:
                # Fixed-length fields
                if shape is not None and len(shape) > 0:
                    # Multi-dimensional array - use fixed_size_list
                    total_size = int(np.prod(shape))
                    arrow_type = pa.list_(base_type, total_size)
                else:
                    # Scalar or will be overridden by fixed_length
                    fixed_length = field_spec.get('length', None)
                    if fixed_length is not None:
                        arrow_type = pa.list_(base_type, fixed_length)
                    else:
                        arrow_type = base_type
            
            fields.append(pa.field(field_name, arrow_type))
        
        return pa.schema(fields)

    def create_file(self, name, num_samples, rng):
        """
        Create a Parquet file with generated data.
        """
        # Generate data for each field
        data_dict = {}
        for field_name, field_spec in self.field_specs.items():
            data_dict[field_name] = self._generate_field_data(field_name, field_spec, num_samples, rng)
        
        # Create PyArrow table
        schema = self._create_arrow_schema()
        
        # Convert data to PyArrow arrays
        arrays = []
        for field_name in self.field_specs.keys():
            data = data_dict[field_name]
            field_spec = self.field_specs[field_name]
            
            # Handle different data types
            if field_spec.get('variable_length', False):
                arrays.append(pa.array(data))
            elif field_spec.get('shape', None):
                if isinstance(data, np.ndarray):
                    flattened = data.reshape(num_samples, -1)
                    arrays.append(pa.array([list(row) for row in flattened]))
                else:
                    arrays.append(pa.array(data))
            else:
                arrays.append(pa.array(data))
        
        table = pa.Table.from_arrays(arrays, schema=schema)
        
        # Write to Parquet file with row group control
        write_kwargs = {
            'compression': self.parquet_compression,
            'use_dictionary': True,
            'write_statistics': True
        }
        
        # Only add compression_level for codecs that support it
        # Codecs that support compression levels: gzip, brotli, zstd
        # Codecs that DON'T support levels: snappy, lz4, none
        codecs_with_levels = {'gzip', 'brotli', 'zstd'}

        if (self.parquet_compression in codecs_with_levels and self.compression_level is not None):
            write_kwargs['compression_level'] = self.compression_level
            
        # Control row group size
        if self.row_group_size is not None:
            # Specify exact row group size
            write_kwargs['row_group_size'] = self.row_group_size
        elif self.num_row_groups_per_file is not None:
            # Calculate row group size based on desired number of groups
            write_kwargs['row_group_size'] = max(1, num_samples // self.num_row_groups_per_file)
        
        pq.write_table(table, name, **write_kwargs)
    @dlp.log
    def generate(self):
        """
        Generate Parquet data for training.
        """
        super().generate()
        
        np.random.seed(10)
        rng = np.random.default_rng()
        
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            progress(i + 1, self.total_files_to_generate, "Generating Parquet Data")
            
            out_path_spec = self.storage.get_uri(self._file_list[i])
            self.create_file(name=out_path_spec, num_samples=self.num_samples, rng=rng)
        
        np.random.seed()
