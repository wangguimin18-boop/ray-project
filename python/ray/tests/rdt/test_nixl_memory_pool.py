"""Unit tests for MemoryPoolManager.

MemoryPoolManager 的单元测试。
"""

import sys

import pytest
import torch

from ray.experimental.rdt.nixl_memory_pool import (
    MemoryPoolManager,
    NixlOutOfMemoryError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# 辅助函数


def _make_tensor(values, dtype=torch.float32):
    """Create a contiguous CPU tensor.

    创建一个连续的 CPU Tensor。
    """
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# allocate_for_tensors — basic allocation and data copy
# ---------------------------------------------------------------------------
# allocate_for_tensors — 基本分配与数据拷贝


class TestAllocateForTensors:
    def test_single_tensor(self):
        t = _make_tensor([1.0, 2.0, 3.0])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([t])

        assert len(views) == 1
        assert torch.equal(views[0], t)
        assert pool.has_block(t)

    def test_multiple_independent_tensors(self):
        t1 = _make_tensor([1.0, 2.0])
        t2 = _make_tensor([3.0, 4.0, 5.0])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([t1, t2])

        assert len(views) == 2
        assert torch.equal(views[0], t1)
        assert torch.equal(views[1], t2)
        assert pool.has_block(t1)
        assert pool.has_block(t2)

    def test_pool_views_are_backed_by_pool_tensor(self):
        """Returned views should be backed by the pool's internal tensor,
        not the source tensor's storage.

        返回的 view 应由池的内部 Tensor 提供存储支持，
        而非源 Tensor 的 storage。
        """
        t = _make_tensor([10.0, 20.0])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([t])

        # The view's storage should be the pool tensor's storage.
        # view 的 storage 应为池 Tensor 的 storage。
        assert (
            views[0].untyped_storage().data_ptr()
            == pool.get_pool_tensor().untyped_storage().data_ptr()
        )

    def test_data_is_copied_not_aliased(self):
        """Mutating the source tensor after allocation should not affect
        the pool copy.

        分配后修改源 Tensor 不应影响池中的拷贝。
        """
        t = _make_tensor([1.0, 2.0, 3.0])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([t])

        original = views[0].clone()
        t[0] = 999.0
        assert torch.equal(views[0], original)


# ---------------------------------------------------------------------------
# allocate_for_tensors — storage deduplication
# ---------------------------------------------------------------------------
# allocate_for_tensors — storage 去重


class TestStorageDeduplication:
    def test_views_of_same_storage_share_one_block(self):
        """Two views of the same underlying storage should produce only one
        pool allocation.

        同一底层 storage 的两个 view 应只产生一次池分配。
        """
        base = _make_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        view_a = base[0:2]
        view_b = base[1:3]

        storage_size = base.untyped_storage().nbytes()
        # Pool is exactly one storage — a second allocation would OOM.
        # 池大小恰好为一个 storage —— 第二次分配会导致 OOM。
        pool = MemoryPoolManager(pool_size=storage_size, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([view_a, view_b])

        assert len(views) == 2
        assert torch.equal(views[0], view_a)
        assert torch.equal(views[1], view_b)

    def test_duplicate_tensor_in_list(self):
        """The exact same tensor object appearing twice should deduplicate.

        同一个 Tensor 对象在列表中出现两次时应被去重。
        """
        t = _make_tensor([1.0, 2.0])
        storage_size = t.untyped_storage().nbytes()
        pool = MemoryPoolManager(pool_size=storage_size, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([t, t])

        assert len(views) == 2
        assert torch.equal(views[0], t)
        assert torch.equal(views[1], t)

    def test_cross_call_reuse(self):
        """A second allocate_for_tensors call with the same tensor should
        reuse the existing pool block (cache hit), not allocate a new one.

        第二次对同一 Tensor 调用 allocate_for_tensors 应复用已有的池 block（缓存命中），
        而不是分配新的 block。
        """
        t = _make_tensor([1.0, 2.0, 3.0])
        storage_size = t.untyped_storage().nbytes()
        # Pool fits exactly one storage.
        # 池大小恰好容纳一个 storage。
        pool = MemoryPoolManager(pool_size=storage_size, device=torch.device("cpu"))

        views1 = pool.allocate_for_tensors([t])
        # Second call should hit cache, not OOM.
        # 第二次调用应命中缓存，不应 OOM。
        views2 = pool.allocate_for_tensors([t])

        assert torch.equal(views1[0], t)
        assert torch.equal(views2[0], t)

    def test_mixed_cache_hit_and_new_allocation(self):
        """One call with a mix of already-allocated and new tensors should
        only allocate for the new ones.

        对已分配和全新 Tensor 混合调用时，应只分配新 Tensor 的 block。
        """
        t1 = _make_tensor([1.0, 2.0])
        t2 = _make_tensor([3.0, 4.0, 5.0])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))

        # Pre-allocate t1.
        # 预分配 t1。
        pool.allocate_for_tensors([t1])

        # Now allocate both — t1 should cache-hit, t2 should get new block.
        # 现在分配两者 — t1 应缓存命中，t2 应获得新 block。
        views = pool.allocate_for_tensors([t1, t2])
        assert len(views) == 2
        assert torch.equal(views[0], t1)
        assert torch.equal(views[1], t2)
        assert pool.has_block(t2)


# ---------------------------------------------------------------------------
# allocate_for_tensors — OOM
# ---------------------------------------------------------------------------
# allocate_for_tensors — 内存不足（OOM）


class TestOOM:
    def test_oom_single_tensor(self):
        t = _make_tensor([1.0, 2.0, 3.0])  # 12 bytes
        # 12 字节
        pool = MemoryPoolManager(pool_size=4, device=torch.device("cpu"))

        with pytest.raises(NixlOutOfMemoryError, match="out of memory"):
            pool.allocate_for_tensors([t])

    def test_oom_does_not_corrupt_pool_state(self):
        """After an OOM error, the pool state should be unchanged — previously
        allocated blocks remain valid and no partial allocation leaks.

        OOM 错误后池状态应不变 —— 之前分配的 block 仍然有效，且无部分分配泄漏。
        """
        t1 = _make_tensor([1.0, 2.0])  # 8 bytes
        t2 = _make_tensor([3.0, 4.0, 5.0])  # 12 bytes
        # t1 8 字节，t2 12 字节
        pool = MemoryPoolManager(pool_size=12, device=torch.device("cpu"))

        views1 = pool.allocate_for_tensors([t1])
        assert torch.equal(views1[0], t1)

        # t2 doesn't fit in the remaining 4 bytes.
        # t2 无法放入剩余的 4 字节空间。
        with pytest.raises(NixlOutOfMemoryError):
            pool.allocate_for_tensors([t2])

        # Pool should still be intact — t1's block is still valid.
        # 池应保持完整 —— t1 的 block 仍然有效。
        assert pool.has_block(t1)

    def test_atomic_allocation_failure(self):
        """When allocating multiple tensors atomically, if one doesn't fit,
        none should be allocated.

        原子分配多个 Tensor 时，若其中一个无法放入，则全部都不应被分配。
        """
        t1 = _make_tensor([1.0])  # 4 bytes
        t2 = _make_tensor([1.0] * 100)  # 400 bytes — won't fit
        # t1 4 字节，t2 400 字节 — 无法放入
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))

        with pytest.raises(NixlOutOfMemoryError):
            pool.allocate_for_tensors([t1, t2])

        # Neither tensor should have been tracked.
        # 两个 Tensor 都不应被跟踪。
        assert not pool.has_block(t1)
        assert not pool.has_block(t2)


# ---------------------------------------------------------------------------
# free_tensors
# ---------------------------------------------------------------------------
# free_tensors（释放 Tensor）


class TestFreeTensors:
    def test_free_and_reallocate(self):
        """After freeing, the space should be reusable.

        释放后，空间应可重新使用。
        """
        t1 = _make_tensor([1.0, 2.0])  # 8 bytes
        # t1 8 字节
        pool = MemoryPoolManager(pool_size=8, device=torch.device("cpu"))

        pool.allocate_for_tensors([t1])
        assert pool.has_block(t1)

        pool.free_tensors([t1])
        assert not pool.has_block(t1)

        # Now a new tensor of the same size should fit.
        # 现在一个相同大小的新 Tensor 应能放入。
        t2 = _make_tensor([3.0, 4.0])
        views = pool.allocate_for_tensors([t2])
        assert torch.equal(views[0], t2)

    def test_free_unknown_tensor_is_noop(self):
        """Freeing a tensor that was never allocated should not raise.

        释放从未被分配的 Tensor 不应抛出异常。
        """
        t = _make_tensor([1.0])
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))
        # Should not raise.
        # 不应抛出异常。
        pool.free_tensors([t])

    def test_free_multiple_tensors(self):
        t1 = _make_tensor([1.0, 2.0])
        t2 = _make_tensor([3.0, 4.0])
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))

        pool.allocate_for_tensors([t1])
        pool.allocate_for_tensors([t2])
        pool.free_tensors([t1, t2])

        assert not pool.has_block(t1)
        assert not pool.has_block(t2)

    def test_free_then_cross_call_reuse_is_broken(self):
        """After freeing, the same tensor should NOT get a cache hit — it
        should allocate a fresh block.

        释放后，同一 Tensor 不应获得缓存命中 —— 应分配一个全新的 block。
        """
        t = _make_tensor([1.0, 2.0])
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))

        pool.allocate_for_tensors([t])
        pool.free_tensors([t])
        assert not pool.has_block(t)

        # Re-allocate — should work (fresh allocation, not cache hit).
        # 重新分配 —— 应成功（全新分配，非缓存命中）。
        views = pool.allocate_for_tensors([t])
        assert torch.equal(views[0], t)
        assert pool.has_block(t)

    def test_double_free_is_noop(self):
        """Freeing an already-freed tensor should not raise or corrupt state.

        重复释放已释放的 Tensor 不应抛出异常或破坏状态。
        """
        t = _make_tensor([1.0, 2.0])
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))

        pool.allocate_for_tensors([t])
        pool.free_tensors([t])
        # Second free — should be a no-op.
        # 第二次释放 —— 应为空操作。
        pool.free_tensors([t])
        assert not pool.has_block(t)


# ---------------------------------------------------------------------------
# Block merging — allocation succeeds only after freed blocks are coalesced
# ---------------------------------------------------------------------------
# Block 合并 —— 仅在释放的 block 被合并后分配才能成功


class TestBlockMerging:
    def test_allocation_requires_merged_free_space(self):
        """After freeing adjacent blocks, the merged space should be usable
        for a single large allocation that wouldn't fit in either fragment.

        释放相邻 block 后，合并的空间应可用于一次大型分配，
        该分配无法放入任一单独的碎片中。
        """
        # Pool: 24 bytes, allocate three 8-byte tensors to fill it.
        # 池：24 字节，分配三个 8 字节 Tensor 以填满。
        t1 = _make_tensor([1.0, 2.0])  # 8 bytes
        t2 = _make_tensor([3.0, 4.0])  # 8 bytes
        t3 = _make_tensor([5.0, 6.0])  # 8 bytes
        # t1/t2/t3 各 8 字节
        pool = MemoryPoolManager(pool_size=24, device=torch.device("cpu"))

        pool.allocate_for_tensors([t1, t2, t3])

        t_big = _make_tensor([7.0, 8.0, 9.0, 10.0])  # 16 bytes
        # t_big 16 字节

        # Free only t1 — 8 bytes free, not enough for t_big (16 bytes).
        # 仅释放 t1 —— 8 字节空闲，不足以放入 t_big（16 字节）。
        pool.free_tensors([t1])
        with pytest.raises(NixlOutOfMemoryError):
            pool.allocate_for_tensors([t_big])

        # Free t2 — now t1+t2 are adjacent and merged into 16 bytes free.
        # 释放 t2 —— 现在 t1+t2 相邻且合并为 16 字节空闲。
        pool.free_tensors([t2])
        views = pool.allocate_for_tensors([t_big])
        assert torch.equal(views[0], t_big)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
# 边界情况


class TestEdgeCases:
    def test_empty_tensor_list(self):
        """allocate_for_tensors with an empty list should return an empty list.

        对空列表调用 allocate_for_tensors 应返回空列表。
        """
        pool = MemoryPoolManager(pool_size=64, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([])
        assert views == []

    def test_different_dtypes(self):
        """Tensors of different dtypes should each get their own block.

        不同 dtype 的 Tensor 应各自获得独立的 block。
        """
        t_f32 = torch.tensor([1.0], dtype=torch.float32)
        t_f64 = torch.tensor([1.0], dtype=torch.float64)
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))

        views = pool.allocate_for_tensors([t_f32, t_f64])
        assert views[0].dtype == torch.float32
        assert views[1].dtype == torch.float64
        assert torch.equal(views[0], t_f32)
        assert torch.equal(views[1], t_f64)

    def test_view_with_storage_offset(self):
        """A tensor view with non-zero storage offset should be correctly
        mapped to the pool.

        具有非零 storage_offset 的 Tensor view 应正确映射到池中。
        """
        base = _make_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        view = base[2:4]  # [3.0, 4.0], storage_offset = 2
        # [3.0, 4.0]，storage_offset = 2

        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        views = pool.allocate_for_tensors([view])

        assert torch.equal(views[0], view)
        assert views[0].shape == (2,)

    def test_multidimensional_tensor_shape_preserved(self):
        """Multi-dimensional tensor shapes should be preserved in pool views.

        多维 Tensor 的形状应在池 view 中保留。
        """
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))

        views = pool.allocate_for_tensors([t])
        assert views[0].shape == (3, 2)
        assert torch.equal(views[0], t)

    def test_allocate_multiple_preserves_request_order(self):
        """_allocate_multiple should return blocks in the same order as the
        input sizes, even though it allocates largest-first internally.

        _allocate_multiple 应按输入 sizes 的顺序返回 block，
        即使内部按最大优先分配。
        """
        pool = MemoryPoolManager(pool_size=1024, device=torch.device("cpu"))
        # Sizes in non-sorted order.
        # sizes 为非排序顺序。
        sizes = [10, 50, 20, 40]
        result = pool._allocate_multiple(sizes)

        assert result is not None
        # Each result block should match the requested size, in order.
        # 每个 result block 应按顺序匹配请求的大小。
        for i, size in enumerate(sizes):
            assert result[i].size == size


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
