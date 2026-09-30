"""Small CPU correctness check; does not import vLLM or native LMCache ops."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

# Load the actual helpers without LMCache's GPU-dependent package initialization.
package = ModuleType("head_test_backend")
package.__path__ = [str(Path(__file__).parents[1] / "LMCache/lmcache/v1/storage_backend")]
sys.modules[package.__name__] = package
heads = importlib.import_module("head_test_backend.head_segments")


class Memory:
    """A reference-counted CPU allocation, with an intentionally tiny budget."""
    def __init__(self, pool, shape, dtype):
        self.pool = pool
        self.tensor = torch.empty(shape, dtype=dtype)
        self.metadata = SimpleNamespace(phy_size=self.tensor.nbytes)
        self.refs = 1
        pool.used += self.tensor.nbytes

    def get_tensor(self, index):
        return self.tensor

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1
        if self.refs == 0:
            self.pool.used -= self.tensor.nbytes


class Pool:
    def __init__(self, capacity):
        self.capacity, self.used = capacity, 0

    def allocate(self, shape, dtype):
        size = torch.Size(shape).numel() * dtype.itemsize
        return Memory(self, shape, dtype) if self.used + size <= self.capacity else None


class HeadSegmentsTest(unittest.TestCase):
    def test_positions_gqa_completeness_and_cache_ownership(self):
        source = torch.arange(2 * 2 * 17 * 8, dtype=torch.float16).reshape(2, 2, 17, 8)
        descriptor, values = heads.split_chunk(source, "parent", "model-revision", 2, 4)
        table = heads.HeadSegmentTable(descriptor)
        self.assertEqual(table.get(1, 0).positions.as_dict(), {"start": 0, "count": 17})
        self.assertEqual(heads.query_to_kv_heads(8, 2, [7, 1, 4]), (1, 0, 1))
        reordered = dict(reversed(list(values.items())))
        self.assertTrue(torch.equal(heads.assemble_chunk(descriptor, reordered), source))
        missing = dict(values)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):
            heads.assemble_chunk(descriptor, missing)
        first, second = list(values)[:2]
        values[first].zero_()
        self.assertTrue(source.any())  # Independent storage, not a full-parent view.
        self.assertTrue(values[second].any())

        size = values[first].nbytes
        pool = Pool(2 * size)
        cache = heads.HeadSegmentCache(size, pool.allocate)
        self.assertTrue(cache.retain(first, values[first]))
        with cache.lease([first]) as resident:
            self.assertTrue(torch.equal(resident[first], values[first]))
            self.assertFalse(cache.remove(first))
            self.assertFalse(cache.retain(second, values[second]))
        self.assertTrue(cache.retain(second, values[second]))
        self.assertEqual(cache.used_bytes, size)
        cache.clear()
        self.assertEqual(pool.used, 0)


if __name__ == "__main__":
    unittest.main()
