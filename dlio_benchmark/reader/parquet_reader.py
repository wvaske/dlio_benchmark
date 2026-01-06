
import numpy as np
import pyarrow.parquet as pq
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow, dft_ai

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Reader for Parquet columnar files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.parquet_file = None
        self.field_specs = getattr(self._args, 'parquet_field_specs', {})
        
        # Cache for row group data to avoid re-reading
        self._row_group_cache = None
        self._cache_row_group_idx = -1
        self._cache_filename = None

    @dlp.log
    def open(self, filename):
        super().open(filename)
        self.parquet_file = pq.ParquetFile(filename)
        self.open_file_map[filename] = self.parquet_file
        return self.parquet_file

    @dlp.log
    def close(self, filename):
        if filename in self.open_file_map:
            # PyArrow ParquetFile doesn't need explicit close
            del self.open_file_map[filename]
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Get a single sample from the parquet file.
        Uses row group caching for efficiency.
        """
        super().get_sample(filename, sample_index)
        parquet_file = self.open_file_map[filename]

        # Determine which columns to read
        columns = [field for field in self.field_specs.keys()
                   if self.field_specs[field].get('read', True)]
        if not columns:
            columns = None  # Read all columns

        # Determine which row group contains this sample
        rows_per_group = parquet_file.metadata.row_group(0).num_rows
        row_group_idx = sample_index // rows_per_group
        row_idx = sample_index % rows_per_group

        # Check if we need to read a new row group
        if (self._cache_filename != filename or 
            self._cache_row_group_idx != row_group_idx):
            
            # Read entire row group and cache it (actual I/O from disk)
            self._row_group_cache = parquet_file.read_row_group(
                row_group_idx, 
                columns=columns
            )
            self._cache_row_group_idx = row_group_idx
            self._cache_filename = filename
        
        # Extract the specific row from cached row group (fast, in-memory)
        row_data = self._row_group_cache.slice(row_idx, 1)

        # Calculate total size from the PyArrow table
        total_size = 0
        for column_name in row_data.column_names:
            column = row_data.column(column_name)
            total_size += column.nbytes

        dlp.update(image_size=total_size)
        dft_ai.update(image_size=total_size)

        # Return the PyArrow table slice
        return row_data

    def next(self):
        """
        Iterator-based reading - delegates to parent class
        which calls get_sample for each sample.
        """
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        """
        Index-based reading - delegates to parent class
        which maps image_idx to filename/sample_index and calls get_sample.
        """
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        # Clear cache
        self._row_group_cache = None
        self._cache_row_group_idx = -1
        self._cache_filename = None
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
