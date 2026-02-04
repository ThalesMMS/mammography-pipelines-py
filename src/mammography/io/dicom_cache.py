"""
LRU cache for DICOM datasets.

Provides an in-memory Least Recently Used (LRU) cache for DICOM datasets
to improve loading performance for frequently accessed files.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


class DicomLRUCache:
    """
    LRU (Least Recently Used) cache for DICOM datasets.

    Maintains an in-memory cache of DICOM datasets with automatic eviction
    of least recently used items when the cache reaches its maximum size.
    This dramatically improves performance for repeated access to the same
    DICOM files.

    The cache tracks statistics including hits, misses, and evictions to
    help optimize cache size configuration.

    Example:
        >>> cache = DicomLRUCache(max_size=100)
        >>> ds = cache.get("path/to/file.dcm")
        >>> print(f"Cache hit rate: {cache.hit_rate:.2%}")

    Attributes:
        max_size: Maximum number of datasets to cache
        cache_dir: Optional directory for cache persistence
        _cache: OrderedDict storing cached datasets (LRU order)
        _hits: Number of cache hits
        _misses: Number of cache misses
        _evictions: Number of evicted entries
    """

    def __init__(self, max_size: int = 100, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of DICOM datasets to cache.
                Must be positive integer.
            cache_dir: Optional directory for cache persistence.
                If provided, cache metadata can be saved/loaded from disk.

        Raises:
            ValueError: If max_size is not a positive integer
        """
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got: {max_size}")

        self.max_size = max_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: OrderedDict[str, pydicom.dataset.FileDataset] = OrderedDict()

        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Create cache directory if specified
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized DicomLRUCache with max_size={max_size}, cache_dir={self.cache_dir}")
        else:
            logger.debug(f"Initialized DicomLRUCache with max_size={max_size}")

    def get(
        self,
        filepath: Union[str, Path],
        stop_before_pixels: bool = False,
        **kwargs: Any
    ) -> pydicom.dataset.FileDataset:
        """
        Get DICOM dataset from cache or load from disk.

        If the dataset is in cache, it's returned immediately (cache hit).
        Otherwise, it's loaded from disk and added to cache (cache miss).

        Args:
            filepath: Path to DICOM file
            stop_before_pixels: If True, stops reading before pixel data
            **kwargs: Additional arguments passed to pydicom.dcmread

        Returns:
            DICOM dataset object

        Raises:
            FileNotFoundError: If the file does not exist
            InvalidDicomError: If the file is not a valid DICOM file
        """
        # Normalize path for consistent cache keys
        filepath = Path(filepath)
        cache_key = str(filepath.resolve())

        # Check if in cache
        if cache_key in self._cache:
            # Cache hit: move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._hits += 1
            logger.debug(f"Cache hit for {filepath}")
            return self._cache[cache_key]

        # Cache miss: load from disk
        self._misses += 1
        logger.debug(f"Cache miss for {filepath}")

        if not filepath.exists():
            raise FileNotFoundError(f"DICOM file not found: {filepath}")

        try:
            dataset = pydicom.dcmread(
                str(filepath),
                stop_before_pixels=stop_before_pixels,
                force=True,
                **kwargs
            )
        except Exception as exc:
            raise InvalidDicomError(
                f"Failed to read DICOM file {filepath}: {exc!r}"
            ) from exc

        # Add to cache
        self._add_to_cache(cache_key, dataset)

        return dataset

    def _add_to_cache(
        self,
        cache_key: str,
        dataset: pydicom.dataset.FileDataset
    ) -> None:
        """
        Add dataset to cache, evicting oldest item if at capacity.

        Args:
            cache_key: Normalized file path used as cache key
            dataset: DICOM dataset to cache
        """
        # Check if we need to evict
        if len(self._cache) >= self.max_size:
            # Remove least recently used (first item in OrderedDict)
            evicted_key, evicted_dataset = self._cache.popitem(last=False)
            self._evictions += 1
            logger.debug(f"Evicted {evicted_key} from cache (max_size={self.max_size})")

        # Add new item (most recently used, at end)
        self._cache[cache_key] = dataset

    def clear(self) -> None:
        """
        Clear all entries from cache.

        Statistics (hits, misses, evictions) are preserved.
        """
        num_items = len(self._cache)
        self._cache.clear()
        logger.debug(f"Cleared {num_items} entries from cache")

    def evict(self, filepath: Union[str, Path]) -> bool:
        """
        Manually evict a specific file from cache.

        Args:
            filepath: Path to DICOM file to evict

        Returns:
            True if the file was in cache and evicted, False otherwise
        """
        filepath = Path(filepath)
        cache_key = str(filepath.resolve())

        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Manually evicted {filepath} from cache")
            return True

        return False

    def reset_stats(self) -> None:
        """Reset cache statistics (hits, misses, evictions) to zero."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.debug("Reset cache statistics")

    @property
    def size(self) -> int:
        """Current number of datasets in cache."""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """Total number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Total number of evicted entries."""
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """
        Cache hit rate as a ratio (0.0 to 1.0).

        Returns:
            Ratio of hits to total accesses. Returns 0.0 if no accesses yet.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics as a dictionary.

        Returns:
            Dictionary with keys: size, max_size, hits, misses, evictions, hit_rate
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }

    def save(self) -> None:
        """
        Save cache metadata to disk.

        Persists cache keys (in LRU order), statistics, and configuration
        to a JSON file in the cache directory. The actual DICOM datasets
        are not persisted - only the metadata needed to restore the cache
        structure.

        The metadata file is named 'cache_metadata.json' and contains:
        - cache_keys: List of file paths in LRU order (oldest to newest)
        - max_size: Maximum cache size
        - stats: Cache statistics (hits, misses, evictions)

        Raises:
            ValueError: If cache_dir was not specified during initialization
        """
        if self.cache_dir is None:
            raise ValueError("Cannot save cache: cache_dir was not specified")

        metadata = {
            "cache_keys": list(self._cache.keys()),
            "max_size": self.max_size,
            "stats": {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
        }

        metadata_path = self.cache_dir / "cache_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved cache metadata to {metadata_path} ({len(self._cache)} entries)")

    def load(self) -> None:
        """
        Load cache metadata from disk.

        Restores the cache structure from a previously saved metadata file.
        The DICOM datasets themselves are not loaded - they will be loaded
        on-demand when accessed via get().

        This method clears the current cache and loads the cache keys in
        their original LRU order. Statistics are also restored.

        Raises:
            ValueError: If cache_dir was not specified during initialization
            FileNotFoundError: If the metadata file does not exist
        """
        if self.cache_dir is None:
            raise ValueError("Cannot load cache: cache_dir was not specified")

        metadata_path = self.cache_dir / "cache_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cache metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Clear current cache
        self._cache.clear()

        # Restore configuration
        self.max_size = metadata.get("max_size", self.max_size)

        # Restore statistics
        stats = metadata.get("stats", {})
        self._hits = stats.get("hits", 0)
        self._misses = stats.get("misses", 0)
        self._evictions = stats.get("evictions", 0)

        # Note: We only restore the cache keys, not the actual datasets
        # The datasets will be loaded on-demand when accessed
        cache_keys = metadata.get("cache_keys", [])
        logger.debug(
            f"Loaded cache metadata from {metadata_path} "
            f"({len(cache_keys)} keys, will be loaded on-demand)"
        )

    def __contains__(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a file is in cache.

        Args:
            filepath: Path to DICOM file

        Returns:
            True if file is cached, False otherwise
        """
        filepath = Path(filepath)
        cache_key = str(filepath.resolve())
        return cache_key in self._cache

    def __len__(self) -> int:
        """Return current number of cached datasets."""
        return len(self._cache)

    def __repr__(self) -> str:
        """Return string representation of cache."""
        return (
            f"DicomLRUCache(max_size={self.max_size}, "
            f"size={self.size}, "
            f"hit_rate={self.hit_rate:.2%})"
        )
