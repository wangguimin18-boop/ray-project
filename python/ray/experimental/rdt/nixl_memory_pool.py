"""Memory pool management for NIXL RDT optimization.

NIXL RDT 优化的内存池管理。"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class NixlOutOfMemoryError(RuntimeError):
    """Raised when the NIXL memory pool runs out of space.

    当 NIXL 内存池空间不足时抛出此异常。

    The pre-allocated memory pool does not have enough free space for the
    requested allocation. Increase the pool size passed to
    ``register_nixl_memory_pool`` to avoid this error.

    预分配的内存池没有足够的可用空间来满足请求的分配。
    增大传递给 ``register_nixl_memory_pool`` 的池大小可避免此错误。
    """


class MemoryBlock:
    """Represents a memory block in the pool.

    表示池中的一个内存块。"""

    def __init__(self, offset: int, size: int):
        self.offset = offset
        self.size = size

    def __repr__(self):
        return f"MemoryBlock(offset={self.offset}, size={self.size})"


class MemoryPoolManager:
    """Manages a pre-allocated memory pool for NIXL RDT transfers.

    本类为 NIXL RDT 传输管理预分配的内存池。

    This class provides a memory allocator interface over a pre-allocated memory pool,
    allowing reuse of registered memory descriptors across multiple transfers.

    本类在预分配的内存池上提供内存分配器接口，
    允许在多次传输中复用已注册的内存描述符。

    It also tracks which storage data pointers have allocated blocks, enabling
    cross-call reuse (the same storage can reuse its pool slot across multiple
    ray.put calls) and pool-level block management.

    本类还跟踪哪些 storage 数据指针已分配了内存块，
    实现跨调用复用（同一 storage 可在多次 ray.put 调用中复用其池槽位）
    以及池级别的内存块管理。
    """

    def __init__(self, pool_size: int, device: "torch.device"):
        """Initialize the memory pool manager.

        初始化内存池管理器。

        Args:
            pool_size: Size of the memory pool in bytes.
            device: Device to allocate the pool on.

        参数:
            pool_size: 内存池的大小（字节）。
            device: 分配内存池的设备。
        """
        import torch

        self.pool_size = pool_size
        self.device = device

        # Allocate the memory pool as a single tensor
        # 将内存池分配为一个单一的 tensor
        # We use a 1D tensor of uint8 to represent raw memory
        # 我们使用 uint8 类型的一维 tensor 来表示原始内存
        self._pool_tensor = torch.zeros(
            pool_size, dtype=torch.uint8, device=self.device
        )

        # Track free blocks using a largest-request-first, first-fit allocator.
        # 使用最大请求优先、首次适配的分配策略来跟踪空闲内存块。
        # List of MemoryBlock for free blocks, sorted by offset.
        # 空闲内存块的 MemoryBlock 列表，按 offset 排序。
        self._free_blocks: List[MemoryBlock] = [MemoryBlock(offset=0, size=pool_size)]

        # Track allocated blocks by storage data pointer.
        # 按 storage 数据指针跟踪已分配的内存块。
        # Maps storage_data_ptr -> MemoryBlock in the pool.
        # 映射 storage_data_ptr -> 池中的 MemoryBlock。
        self._allocated_blocks: Dict[int, MemoryBlock] = {}

    def get_pool_tensor(self) -> "torch.Tensor":
        """Get the underlying pool tensor.

        Returns:
            The pre-allocated tensor representing the memory pool.

        获取底层的池 tensor。

        返回:
            表示内存池的预分配 tensor。
        """
        return self._pool_tensor

    def has_block(self, tensor: "torch.Tensor") -> bool:
        """Check if a tensor has an allocated block in the pool.

        检查某个 tensor 是否在池中拥有已分配的内存块。

        Args:
            tensor: The tensor to check.

        Returns:
            True if the tensor's storage has an allocated block.

        参数:
            tensor: 要检查的 tensor。

        返回:
            如果该 tensor 的 storage 拥有已分配的内存块则返回 True。
        """
        return tensor.untyped_storage().data_ptr() in self._allocated_blocks

    def free_tensors(self, tensors: List["torch.Tensor"]) -> None:
        """Return pool blocks for the given tensors back to the pool.

        将给定 tensor 的池内存块归还到池中。

        The caller is responsible for calling this method on the same tensors that were previously allocated in the pool before those tensors go out of scope.

        调用者有责任在先前在池中分配的 tensor 即将超出作用域之前，
        对这些 tensor 调用此方法。

        Args:
            tensors: Tensors whose pool blocks should be freed.

        参数:
            tensors: 需要释放池内存块的 tensor 列表。
        """
        blocks = []
        for tensor in tensors:
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in self._allocated_blocks:
                blocks.append(self._allocated_blocks.pop(ptr))
        if blocks:
            self._free_multiple(blocks)

    def allocate_for_tensors(
        self, tensors: List["torch.Tensor"]
    ) -> List["torch.Tensor"]:
        """Allocate pool blocks for unique storages, copy data in,
        and return pool-backed tensor views for each input tensor. The caller is responsible for calling free on the original tensors to return the allocated tensor views back to the pool before the original tensors go out of scope.

        为唯一的 storage 分配池内存块，将数据拷贝进去，
        并为每个输入 tensor 返回由池支持的 tensor 视图。
        调用者有责任在原始 tensor 超出作用域之前，
        对原始 tensor 调用 free 方法，将分配的 tensor 视图归还到池中。

        Handles storage-level deduplication: views of the same storage share
        one pool block within a single call, and the same storage reuses its
        existing pool slot across calls.

        处理 storage 级别的去重：同一 storage 的多个视图在单次调用中共享一个池内存块，
        同一 storage 在跨调用时复用其已有的池槽位。

        Args:
            tensors: Source tensors to allocate pool memory for.

        Returns:
            List of pool-backed tensor views, one per input tensor,
            in the same order.

        Raises:
            NixlOutOfMemoryError: If the pool has insufficient space.

        参数:
            tensors: 需要分配池内存的源 tensor 列表。

        返回:
            由池支持的 tensor 视图列表，每个输入 tensor 对应一个，
            顺序与输入一致。

        异常:
            NixlOutOfMemoryError: 如果池空间不足。
        """
        new_allocations = None
        newly_tracked_ptrs: List[int] = []
        try:
            import torch

            # Deduplicate storages: group tensors by storage data_ptr so
            # views of the same storage share one pool allocation.
            # 对 storage 去重：按 storage data_ptr 分组 tensor，
            # 使得同一 storage 的多个视图共享一次池分配。
            # Maps storage data_ptr -> index in alloc_sizes/new_allocations,
            # or -1 for storages that already have a pool block (cache hit).
            # 映射 storage data_ptr -> alloc_sizes/new_allocations 中的索引，
            # 或 -1 表示已有池内存块的 storage（缓存命中）。
            storage_idx: Dict[int, int] = {}
            # Maps storage data_ptr -> a representative tensor (for copy).
            # 映射 storage data_ptr -> 一个代表性 tensor（用于拷贝）。
            ptr_to_tensor: Dict[int, "torch.Tensor"] = {}
            alloc_sizes: List[int] = []

            for tensor in tensors:
                ptr = tensor.untyped_storage().data_ptr()
                if ptr in storage_idx:
                    continue
                ptr_to_tensor[ptr] = tensor
                if self.has_block(tensor):
                    storage_idx[ptr] = -1
                else:
                    storage_idx[ptr] = len(alloc_sizes)
                    alloc_sizes.append(tensor.untyped_storage().nbytes())

            # Allocate new (non-cached) storages atomically.
            # 原子性地分配新的（非缓存的）storage。
            if alloc_sizes:
                new_allocations = self._allocate_multiple(alloc_sizes)
                if new_allocations is None:
                    raise NixlOutOfMemoryError(
                        f"NIXL memory pool out of memory: cannot allocate "
                        f"{len(alloc_sizes)} block(s) totaling "
                        f"{sum(alloc_sizes)} bytes. Consider increasing "
                        f"the pool size when calling "
                        f"register_nixl_memory_pool."
                    )

            # Track and copy newly allocated blocks. Cache hits keep the
            # originally copied data -- any mutations to the source storage
            # since the first ray.put are not reflected in outstanding refs.
            # 跟踪并拷贝新分配的内存块。缓存命中保留最初拷贝的数据——
            # 自第一次 ray.put 之后对源 storage 的任何修改不会反映在未完成的引用中。
            for ptr, idx in storage_idx.items():
                if idx < 0:
                    continue
                blk = new_allocations[idx]
                self._allocated_blocks[ptr] = blk
                newly_tracked_ptrs.append(ptr)
                # Copy the tensor's full underlying storage into the pool block.
                # 将 tensor 的完整底层 storage 拷贝到池内存块中。
                src = ptr_to_tensor[ptr]
                storage_size = src.untyped_storage().nbytes()
                storage_bytes = torch.tensor(
                    [], dtype=torch.uint8, device=src.device
                ).set_(src.untyped_storage())
                self._pool_tensor[blk.offset : blk.offset + storage_size].copy_(
                    storage_bytes
                )

            # Build pool-backed tensor views for each input tensor.
            # 为每个输入 tensor 构建由池支持的 tensor 视图。
            pool_views: List["torch.Tensor"] = []
            for tensor in tensors:
                ptr = tensor.untyped_storage().data_ptr()
                blk = self._allocated_blocks[ptr]
                pool_offset = blk.offset + (
                    tensor.storage_offset() * tensor.element_size()
                )
                view_byte_size = tensor.numel() * tensor.element_size()
                pool_bytes = self._pool_tensor[
                    pool_offset : pool_offset + view_byte_size
                ]
                pool_views.append(pool_bytes.view(tensor.dtype).reshape(tensor.shape))

            return pool_views

        except Exception:
            # Roll back any pool mutations made in this call, then re-raise.
            # 回滚本次调用中对池所做的任何修改，然后重新抛出异常。
            try:
                if new_allocations is not None:
                    self._free_multiple(new_allocations)
                for ptr in newly_tracked_ptrs:
                    self._allocated_blocks.pop(ptr, None)
            except Exception as cleanup_err:
                logger.error(f"Memory pool cleanup failed: {cleanup_err}.")
            raise

    def _allocate_multiple(self, sizes: List[int]) -> Optional[List[MemoryBlock]]:
        """Allocate multiple memory blocks from the pool atomically.

        Either all allocations succeed, or none of them do.

        从池中原子性地分配多个内存块。

        所有分配要么全部成功，要么全部失败。

        Args:
            sizes: List of sizes to allocate in bytes.

        Returns:
            List of MemoryBlock if all allocations succeed, None otherwise.

        参数:
            sizes: 要分配的大小列表（字节）。

        返回:
            如果所有分配成功则返回 MemoryBlock 列表，否则返回 None。
        """
        if not sizes or any(s <= 0 for s in sizes):
            raise ValueError("Invalid allocation request")

        # If total free space is less than total requested, fail fast.
        # 如果总可用空间小于总请求空间，立即失败。
        total_requested = sum(sizes)
        total_free = sum(b.size for b in self._free_blocks)
        if total_free < total_requested:
            return None

        # Allocate largest first to reduce fragmentation; then return in original order.
        # 先分配最大的以减少碎片；然后按原始顺序返回。
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
        sorted_sizes = [sizes[i] for i in order]

        # Try to allocate all blocks atomically.
        # 尝试原子性地分配所有内存块。
        allocations: List[MemoryBlock] = []
        temp_free_blocks = [MemoryBlock(b.offset, b.size) for b in self._free_blocks]

        for size in sorted_sizes:
            allocated = False
            for i, block in enumerate(temp_free_blocks):
                if block.size >= size:
                    # Allocate at the start of the current free block
                    # 在当前空闲内存块的起始位置分配
                    offset = block.offset
                    remaining_after = block.size - size

                    if remaining_after == 0:
                        temp_free_blocks.pop(i)
                    else:
                        block.offset = offset + size
                        block.size = remaining_after

                    allocations.append(MemoryBlock(offset, size))
                    allocated = True
                    break

            if not allocated:
                # If any size cannot be allocated, the entire batch fails,
                # do not modify the real state.
                # 如果任何大小无法分配，整个批次失败，
                # 不修改真实状态。
                return None

        # Reorder allocations back to original request order
        # 将分配结果重新排序回原始请求顺序
        result: List[MemoryBlock] = [MemoryBlock(0, 0)] * len(sizes)
        for k, alloc in enumerate(allocations):
            result[order[k]] = alloc

        # All successful, submit modifications
        # 全部成功，提交修改
        temp_free_blocks.sort(key=lambda b: b.offset)
        self._free_blocks = temp_free_blocks

        return result

    def _free_multiple(self, blocks: List[MemoryBlock]) -> None:
        """Free multiple memory blocks back to the pool.

        将多个内存块归还到池中。

        Args:
            blocks: Memory blocks to free.

        参数:
            blocks: 要释放的内存块列表。
        """
        if not blocks:
            raise ValueError("Invalid free request")
        self._free_blocks.extend(blocks)

        # Single pass: merge all adjacent free blocks
        # 单次遍历：合并所有相邻的空闲内存块
        self._free_blocks.sort(key=lambda b: b.offset)
        i = 0
        while i < len(self._free_blocks) - 1:
            curr = self._free_blocks[i]
            next_block = self._free_blocks[i + 1]
            if curr.offset + curr.size == next_block.offset:
                curr.size += next_block.size
                self._free_blocks.pop(i + 1)
            else:
                i += 1
