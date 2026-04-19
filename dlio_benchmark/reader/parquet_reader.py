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

Optimized Parquet Reader for DLIO Benchmark.

Key optimizations:
- Memory-mapped file access for reduced syscall overhead
- Metadata caching to avoid repeated footer reads
- Row group caching with LRU eviction
- Column projection to read only needed columns
- Threading support for parallel column decompression
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Optimized Parquet reader for local/network filesystems.

    Uses row-group-granular access with caching. Opens files with memory
    mapping for efficient access. Row groups are cached with LRU eviction.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        # Get parquet-specific configuration
        parquet_config = getattr(self._args, 'parquet', {}) or {}

        # Memory mapping (enabled by default for performance)
        self._use_mmap = parquet_config.get('memory_map', True)

        # Threading for parallel column reads
        self._use_threads = parquet_config.get('use_threads', True)

        # Column selection - list of column names to read (None = all)
        columns_config = parquet_config.get('columns', [])
        if columns_config:
            # Extract column names where read=True
            self._columns = [
                c.get('name') for c in columns_config
                if c.get('read', True)
            ]
            if not self._columns:
                self._columns = None  # Read all if no columns specified
        else:
            self._columns = None

        # Row group cache configuration
        self._rg_cache_size = int(parquet_config.get('row_group_cache_size', 4))
        self._rg_cache = {}  # (filename, rg_idx) -> compressed_bytes
        self._rg_lru = []    # LRU key list

        # Metadata cache to avoid repeated footer reads
        self._metadata_cache = {}  # filename -> (ParquetFile, offsets)
        self._use_metadata_cache = parquet_config.get('metadata_cache', True)

        self.logger.info(
            f"{utcnow()} ParquetReader thread={thread_index} epoch={epoch} "
            f"mmap={self._use_mmap} threads={self._use_threads} "
            f"columns={self._columns} rg_cache_size={self._rg_cache_size}"
        )

    def _evict_lru(self):
        """Evict the least-recently-used row group from cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    @dlp.log
    def open(self, filename):
        """
        Open a Parquet file and read its footer metadata.

        Returns (ParquetFile, cumulative_offsets) where offsets[i] is the
        first row index of row group i, and offsets[-1] is total row count.
        """
        # Check metadata cache first
        if self._use_metadata_cache and filename in self._metadata_cache:
            return self._metadata_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename, memory_map=self._use_mmap)
        meta = pf.metadata

        # Build cumulative row offsets for bisect lookup
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReader.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )

        result = (pf, offsets)

        # Cache metadata if enabled
        if self._use_metadata_cache:
            self._metadata_cache[filename] = result

        return result

    @dlp.log
    def close(self, filename):
        """Evict cached row groups for this file."""
        keys_to_remove = [k for k in self._rg_cache if k[0] == filename]
        for k in keys_to_remove:
            self._rg_cache.pop(k, None)
            if k in self._rg_lru:
                self._rg_lru.remove(k)
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the row group containing sample_index.

        Uses bisect for O(log N) row group lookup. Caches row groups
        with LRU eviction. Reports compressed bytes to profiler.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: offsets[rg_idx] <= sample_index < offsets[rg_idx+1]
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)

        if cache_key not in self._rg_cache:
            # Read row group from disk
            pf.read_row_group(
                rg_idx,
                columns=self._columns,
                use_threads=self._use_threads
            )

            # Calculate compressed size for metrics
            rg_meta = pf.metadata.row_group(rg_idx)
            compressed_bytes = sum(
                rg_meta.column(c).total_compressed_size
                for c in range(rg_meta.num_columns)
            )

            # LRU eviction
            while len(self._rg_cache) >= self._rg_cache_size:
                self._evict_lru()

            self._rg_cache[cache_key] = compressed_bytes
            self._rg_lru.append(cache_key)
        else:
            # Move to end of LRU list (most recently used)
            try:
                self._rg_lru.remove(cache_key)
            except ValueError:
                pass
            self._rg_lru.append(cache_key)

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self._rg_cache.clear()
        self._rg_lru.clear()
        self._metadata_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
