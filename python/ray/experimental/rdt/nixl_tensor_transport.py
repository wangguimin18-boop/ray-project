import logging
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import ray
from ray._private.ray_constants import (
    NIXL_REMOTE_AGENT_CACHE_MAXSIZE,
)
from ray.experimental.rdt.nixl_memory_pool import MemoryPoolManager
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    FetchRequest,
    TensorTransportManager,
    TensorTransportMetadata,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class NixlCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the NIXL communicator.

    NIXL 通信器的元数据。"""


@dataclass
class NixlTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for NIXL transport.

    存储在 GPU 对象存储中用于 NIXL 传输的 Tensor 元数据。

    Args:
        nixl_serialized_descs: Serialized tensor descriptors for NIXL transport.
        nixl_agent_meta: The additional metadata of the remote NIXL agent.
        nixl_agent_name: The name of the NIXL agent.
        nixl_agent_meta_version: The version of the NIXL agent metadata.

    参数：
        nixl_serialized_descs: 用于 NIXL 传输的序列化 Tensor 描述符。
        nixl_agent_meta: 远程 NIXL agent 的附加元数据。
        nixl_agent_name: NIXL agent 的名称。
        nixl_agent_meta_version: NIXL agent 元数据的版本。
    """

    nixl_serialized_descs: Optional[bytes] = None
    nixl_agent_meta: Optional[bytes] = None
    nixl_agent_name: Optional[str] = None
    nixl_agent_meta_version: Optional[int] = 0

    __eq__ = object.__eq__
    __hash__ = object.__hash__


@dataclass
class TensorDesc:
    # nixlRegDList handle, or None for pool-managed tensors (pool memory is
    # registered once at pool creation, so individual tensors don't need their
    # own NIXL registration).
    # nixlRegDList 句柄，对于池管理的 Tensor 为 None（池内存
    # 在池创建时一次性注册，因此单个 Tensor 不需要自己的 NIXL 注册）。
    reg_desc: Any
    # tracks the number of NIXL metadata containing the tensor.
    # 跟踪包含该 Tensor 的 NIXL 元数据数量。
    metadata_count: int


@dataclass
class NixlFetchRequest(FetchRequest):
    """NIXL-specific FetchRequest carrying the async transfer state.

    携带异步传输状态的 NIXL 专用 FetchRequest。

    Returned by fetch_multiple_tensors and consumed by wait_fetch_complete.

    由 fetch_multiple_tensors 返回，由 wait_fetch_complete 消费。

    Args:
        obj_id: Inherited. The object ID for the transfer, used for abort checks and cleanup.
        tensors: Inherited. Pre-allocated output tensors (populated before the transfer starts).
        xfer_handle: NIXL transfer request handle.
        nixl_agent: Reference to the NIXL agent.
        remote_name: Name of the remote NIXL agent.
        remove_tensor_descs: Whether to remove tensor descriptors from the cache during cleanup.

    参数：
        obj_id: 继承。传输的对象 ID，用于中止检查和清理。
        tensors: 继承。预分配的输出 Tensor（在传输开始前填充）。
        xfer_handle: NIXL 传输请求句柄。
        nixl_agent: NIXL agent 的引用。
        remote_name: 远程 NIXL agent 的名称。
        remove_tensor_descs: 清理时是否从缓存中移除 Tensor 描述符。
    """

    xfer_handle: Any = None
    nixl_agent: Any = None
    remote_name: Optional[str] = None
    remove_tensor_descs: bool = False
    transport: Any = None

    def __del__(self):
        if self.transport is not None:
            self.transport._cleanup_transfer(
                self.obj_id,
                self.tensors,
                self.xfer_handle,
                self.remote_name,
                self.remove_tensor_descs,
            )


class NixlTensorTransport(TensorTransportManager):
    def __init__(self):
        # This is lazily initialized because it requires NIXL to actually be installed and we want to allow an owner that is just coordinating to not need to have NIXL installed.
        # 延迟初始化，因为它要求 NIXL 实际安装，并且我们希望仅做协调的 owner 不需要安装 NIXL。
        self._nixl_agent = None
        self._aborted_transfer_obj_ids = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()
        # Mapping from tensor storage data pointer to the NIXL descriptor and reference count.
        # Unlike _managed_meta_nixl, we only deregister tensors when ALL metadata containing the tensor is freed.
        # For pool-managed tensors, reg_desc is None and the pool block is returned instead of deregistering.
        # 从 Tensor 存储数据指针到 NIXL 描述符和引用计数的映射。
        # 与 _managed_meta_nixl 不同，只有当包含该 Tensor 的所有元数据被释放时才注销 Tensor。
        # 对于池管理的 Tensor，reg_desc 为 None，池内存块被归还而不是注销。
        self._tensor_desc_cache: Dict[int, TensorDesc] = {}
        # Mapping from object ID to the NIXL managed meta.
        # The lifetime of _managed_meta_nixl is tied to the object ref and freed when the ref goes out of scope.
        # 从对象 ID 到 NIXL 管理元数据的映射。
        # _managed_meta_nixl 的生命周期与 ObjectRef 绑定，当引用超出作用域时被释放。
        self._managed_meta_nixl: Dict[str, Any] = {}
        # Lock protecting _tensor_desc_cache and _managed_meta_nixl since they can be
        # accessed from the main task execution thread or the _ray_system thread.
        # 保护 _tensor_desc_cache 和 _managed_meta_nixl 的锁，因为它们可以从
        # 主任务执行线程或 _ray_system 线程访问。
        self._cache_lock = threading.RLock()
        # LRU cache of remote agent names. When full, the least
        # recently used remote agent is evicted and remove_remote_agent is called.
        # 远程 agent 名称的 LRU 缓存。当缓存满时，最久未使用的远程 agent
        # 被移除，并调用 remove_remote_agent。
        self._remote_agents: OrderedDict = OrderedDict()
        # Increment the version whenever memory is deregistered.
        # 每次内存注销时递增版本号。
        self._nixl_agent_meta_version = 0
        self._memory_pool: Optional[MemoryPoolManager] = None

    def tensor_transport_backend(self) -> str:
        return "NIXL"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return True

    def register_nixl_memory(self, tensor: "torch.Tensor") -> None:
        """Registers the tensor's memory with NIXL and bumps the reference count so the memory region is never deregistered.

        将 Tensor 的内存注册到 NIXL 并增加引用计数，使内存区域永远不会被注销。
        """
        self._add_tensor_descs([tensor])

    def register_nixl_memory_pool(self, size: int, device: "torch.device") -> None:
        """Pre-allocates a memory pool and registers it with NIXL.

        预分配内存池并将其注册到 NIXL。

        Args:
            size: Size of the memory pool in bytes.
            device: Device to allocate the pool on (cpu or cuda).

        Raises:
            ValueError: If a memory pool is already registered.

        参数：
            size: 内存池的大小，以字节为单位。
            device: 分配内存池的设备（cpu 或 cuda）。

        异常：
            ValueError: 如果已注册了内存池。
        """
        if self._memory_pool is not None:
            raise ValueError(
                "A memory pool is already registered. "
                "Only one memory pool is supported."
            )
        nixl_agent = self.get_nixl_agent()
        pool = MemoryPoolManager(pool_size=size, device=device)
        nixl_agent.register_memory(pool.get_pool_tensor())
        self._memory_pool = pool

    def deregister_nixl_memory(self, tensor: "torch.Tensor") -> None:
        """Decrements the reference count for the tensor's NIXL memory registration.
        If the count reaches 0, the memory is deregistered from NIXL.

        递减 Tensor 的 NIXL 内存注册引用计数。
        如果计数达到 0，则从 NIXL 注销该内存。
        """
        self._remove_tensor_descs([tensor])

    def get_nixl_agent(self):
        """
        Creates a NIXL agent with UCX backend if not already created.

        如果尚未创建，则创建一个带有 UCX 后端的 NIXL agent。
        """
        if self._nixl_agent is not None:
            return self._nixl_agent

        from nixl._api import nixl_agent, nixl_agent_config

        agent_config = nixl_agent_config(backends=["UCX"])
        ctx = ray.get_runtime_context()
        actor_id = ctx.get_actor_id()
        if actor_id is None:
            # If the actor id is None, it means the current process is a driver.
            # 如果 actor id 为 None，表示当前进程是 driver。
            import uuid

            actor_id = f"RAY-DRIVER-{uuid.uuid4()}"
        self._nixl_agent = nixl_agent(actor_id, agent_config)

        return self._nixl_agent

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        # TODO(dayshah): This is called on a .remote RDT call, so it's quite expensive.
        # TODO(dayshah): 这是在 .remote RDT 调用上执行的，因此开销较大。
        def __ray_actor_has_tensor_transport__(
            self: "ray.actor.ActorHandle",
        ) -> bool:
            # Check if nixl is installed
            # 检查 nixl 是否已安装
            try:
                from ray.experimental.rdt.util import (
                    get_tensor_transport_manager,
                )

                get_tensor_transport_manager("NIXL").get_nixl_agent()
                return True
            except Exception:
                return False

        return ray.get(
            actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __ray_actor_has_tensor_transport__
            )
        )

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        rdt_object: List["torch.Tensor"],
    ) -> NixlTransportMetadata:
        import torch

        with self._cache_lock:
            device = None
            tensor_meta = []

            if rdt_object:
                # We assume all tensors in one RDT object have the same device type,
                # but we don't assume they're all on the same device.
                # 我们假设一个 RDT 对象中的所有 Tensor 具有相同的设备类型，
                # 但不假设它们都在同一设备上。
                devices = set()
                device = rdt_object[0].device
                for t in rdt_object:
                    if t.device.type != device.type:
                        raise ValueError(
                            "All tensors in an RDT object must have the same device type."
                        )
                    if not t.is_contiguous():
                        raise ValueError(
                            "All tensors in an RDT object must be contiguous."
                        )
                    tensor_meta.append((t.shape, t.dtype))
                    devices.add(t.device)
                if device.type == "cuda":
                    # We have to synchronize before memory registration to assure the
                    # object has been created because nixl doesn't guarantee it will.
                    # 我们必须在内存注册之前同步，以确保对象已被创建，
                    # 因为 nixl 不保证这一点。
                    for dev in devices:
                        torch.cuda.synchronize(dev)

                nixl_agent = self.get_nixl_agent()
                # Use the pool only when every tensor lives on the exact same
                # device as the pool, AND no tensor already has an existing
                # NIXL registration (via register_nixl_memory).
                # 只有当每个 Tensor 都位于与池完全相同的设备上，
                # 且没有 Tensor 已有现有的 NIXL 注册（通过 register_nixl_memory）时才使用池。
                pool_eligible = (
                    self._memory_pool is not None
                    and all(
                        t.device == self._memory_pool.get_pool_tensor().device
                        for t in rdt_object
                    )
                    and not any(self._tensor_memory_registered(t) for t in rdt_object)
                )
                if pool_eligible:
                    xfer_descs = self._allocate_pool_xfer_descs(rdt_object)
                else:
                    self._add_tensor_descs(rdt_object)
                    xfer_descs = nixl_agent.get_xfer_descs(rdt_object)

                serialized_descs = nixl_agent.get_serialized_descs(xfer_descs)
                agent_meta = nixl_agent.get_agent_metadata()
                agent_name = nixl_agent.name
                agent_meta_version = self._nixl_agent_meta_version
            else:
                serialized_descs, agent_meta = None, None
                agent_name, agent_meta_version = None, None

            ret = NixlTransportMetadata(
                tensor_meta=tensor_meta,
                tensor_device=device.type if device else None,
                nixl_serialized_descs=serialized_descs,
                nixl_agent_meta=agent_meta,
                nixl_agent_name=agent_name,
                nixl_agent_meta_version=agent_meta_version,
            )
            self._put_meta(obj_id, ret)
            return ret

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> NixlCommunicatorMetadata:
        return NixlCommunicatorMetadata()

    def fetch_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> NixlFetchRequest:
        """Initiates an async transfer for multiple tensors.

        This triggers the transfer but does not wait for completion.
        Call wait_fetch_complete(fetch_request) to wait for the transfer to
        finish and retrieve the tensors.

        Args:
            obj_id: The object ID for the transfer.
            tensor_transport_metadata: Metadata for the tensor transport.
            communicator_metadata: Metadata for the communicator.
            target_buffers: Optional pre-allocated buffers to receive tensors into.

        Returns:
            A NixlFetchRequest carrying the async transfer state.

        启动多个 Tensor 的异步传输。

        此方法触发传输但不等待完成。
        调用 wait_fetch_complete(fetch_request) 以等待传输完成并获取 Tensor。

        参数：
            obj_id: 传输的对象 ID。
            tensor_transport_metadata: Tensor 传输的元数据。
            communicator_metadata: 通信器的元数据。
            target_buffers: 可选的预分配缓冲区，用于接收 Tensor。

        返回：
            携带异步传输状态的 NixlFetchRequest。
        """
        from ray.experimental.rdt.util import (
            create_empty_tensors_from_metadata,
        )

        tensors = target_buffers or create_empty_tensors_from_metadata(
            tensor_transport_metadata
        )

        assert isinstance(tensor_transport_metadata, NixlTransportMetadata)
        assert isinstance(communicator_metadata, NixlCommunicatorMetadata)

        nixl_serialized_descs = tensor_transport_metadata.nixl_serialized_descs
        remote_nixl_agent_meta = tensor_transport_metadata.nixl_agent_meta

        with self._aborted_transfer_obj_ids_lock:
            if obj_id in self._aborted_transfer_obj_ids:
                self._aborted_transfer_obj_ids.remove(obj_id)
                raise RuntimeError(f"NIXL transfer aborted for object id: {obj_id}")

        remote_name = None
        xfer_handle = None
        added_tensor_descs = False

        assert tensors

        try:
            nixl_agent = self.get_nixl_agent()
            remote_xfer_descs = nixl_agent.deserialize_descs(nixl_serialized_descs)
            # This creates a placeholder for the tensor in the tensor_desc_cache even though it doesn't have an object ref for caching purposes.
            # 这在 tensor_desc_cache 中为 Tensor 创建了一个占位符，即使它没有用于缓存目的的 ObjectRef。
            self._add_tensor_descs(tensors)
            added_tensor_descs = True
            local_xfer_descs = nixl_agent.get_xfer_descs(tensors)

            remote_name = tensor_transport_metadata.nixl_agent_name
            remote_agent_meta_version = (
                tensor_transport_metadata.nixl_agent_meta_version
            )

            # Nixl agent reuse is enabled.
            # NIXL agent 重用已启用。
            if NIXL_REMOTE_AGENT_CACHE_MAXSIZE > 0:
                if remote_name in self._remote_agents:
                    # If the remote agent metadata version is different from the cached one,
                    # it means there was memory deregistered. We need to remove the remote agent
                    # before adding it, because `nixlRemoteSection` currently does not support
                    # updating descriptor list in such a case (there is potential memory overlap).
                    # 如果远程 agent 元数据版本与缓存版本不同，
                    # 说明有内存被注销了。我们需要在添加之前移除远程 agent，
                    # 因为 `nixlRemoteSection` 当前不支持在这种情况下更新描述符列表（可能存在内存重叠）。
                    if remote_agent_meta_version != self._remote_agents[remote_name]:
                        nixl_agent.remove_remote_agent(remote_name)
                    self._remote_agents.move_to_end(remote_name)
                elif len(self._remote_agents) >= NIXL_REMOTE_AGENT_CACHE_MAXSIZE:
                    evicted_agent_name, _ = self._remote_agents.popitem(last=False)
                    nixl_agent.remove_remote_agent(evicted_agent_name)

                self._remote_agents[remote_name] = remote_agent_meta_version

            nixl_agent.add_remote_agent(remote_nixl_agent_meta)

            xfer_handle = nixl_agent.initialize_xfer(
                "READ",
                local_xfer_descs,
                remote_xfer_descs,
                remote_name,
                b"UUID",
            )

            state = nixl_agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("NIXL transfer got to Error state.")

            return NixlFetchRequest(
                tensors=tensors,
                obj_id=obj_id,
                xfer_handle=xfer_handle,
                nixl_agent=nixl_agent,
                remote_name=remote_name,
                remove_tensor_descs=added_tensor_descs,
                transport=self,
            )
        except Exception:
            self._cleanup_transfer(
                obj_id, tensors, xfer_handle, remote_name, added_tensor_descs
            )
            # TODO(swang): There is a circular import error because ray.util
            # currently depends on ray.experimental.internal_kv.
            # TODO(swang): 存在循环导入错误，因为 ray.util
            # 目前依赖于 ray.experimental.internal_kv。
            from ray.exceptions import RayDirectTransportError

            raise RayDirectTransportError(
                f"The NIXL transfer failed for object id: {obj_id}. The source actor may have died during the transfer. "
                f"The exception thrown from nixl transfer was:\n {traceback.format_exc()}"
            ) from None

    def wait_fetch_complete(
        self, fetch_request: FetchRequest, timeout: float = -1
    ) -> List["torch.Tensor"]:
        """Waits for a previously initiated fetch to complete and returns the tensors.

        Args:
            fetch_request: The NixlFetchRequest returned by fetch_multiple_tensors.
            timeout: Maximum time in seconds to wait. -1 means wait indefinitely.
                0 means return immediately if not ready.

        Returns:
            List of tensors that were transferred.

        Raises:
            RayDirectTransportError: If the transfer failed.
            TimeoutError: If the timeout is exceeded.

        等待先前发起的获取操作完成并返回 Tensor。

        参数：
            fetch_request: fetch_multiple_tensors 返回的 NixlFetchRequest。
            timeout: 最大等待时间（秒）。-1 表示无限等待。
                0 表示如果未就绪则立即返回。

        返回：
            已传输的 Tensor 列表。

        异常：
            RayDirectTransportError: 如果传输失败。
            TimeoutError: 如果超出超时时间。
        """
        assert isinstance(fetch_request, NixlFetchRequest)
        obj_id = fetch_request.obj_id

        if not fetch_request.tensors:
            return fetch_request.tensors

        try:
            # Check the state of the transfer continuously.
            # 持续检查传输状态。
            deadline = None if timeout < 0 else time.monotonic() + timeout
            while True:
                state = self.get_nixl_agent().check_xfer_state(
                    fetch_request.xfer_handle
                )
                if state == "ERR":
                    raise RuntimeError("NIXL transfer got to Error state.")
                if state == "PROC":
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"NIXL transfer timed out after {timeout}s for object id: {obj_id}"
                        )
                    with self._aborted_transfer_obj_ids_lock:
                        if obj_id in self._aborted_transfer_obj_ids:
                            self._aborted_transfer_obj_ids.remove(obj_id)
                            raise RuntimeError(
                                f"NIXL transfer aborted for object id: {obj_id}"
                            )
                    time.sleep(0.001)  # Avoid busy waiting
                    # 避免忙等待
                elif state == "DONE":
                    break

            return fetch_request.tensors
        except TimeoutError:
            raise
        except Exception:
            from ray.exceptions import RayDirectTransportError

            raise RayDirectTransportError(
                f"The NIXL transfer failed for object id: {obj_id}. The source actor may have died during the transfer. "
                f"The exception thrown from nixl transfer was:\n {traceback.format_exc()}"
            ) from None

    def _cleanup_transfer(
        self,
        obj_id: str,
        tensors: List["torch.Tensor"],
        xfer_handle: Any,
        remote_name: Optional[str],
        remove_tensor_descs: bool,
    ) -> None:
        """Cleans up resources after a transfer completes or fails.

        传输完成或失败后清理资源。
        """
        # We could raise errors or NIXL could raise errors like NIXL_ERR_REMOTE_DISCONNECT,
        # so doing best effort cleanup.
        # 我们可能抛出错误，或 NIXL 可能抛出类似 NIXL_ERR_REMOTE_DISCONNECT 的错误，
        # 因此进行尽力清理。
        nixl_agent = self._nixl_agent
        if nixl_agent is None:
            return
        # We could raise errors or NIXL could raise errors like NIXL_ERR_REMOTE_DISCONNECT,
        # so doing best effort cleanup.
        # 我们可能抛出错误，或 NIXL 可能抛出类似 NIXL_ERR_REMOTE_DISCONNECT 的错误，
        # 因此进行尽力清理。
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.discard(obj_id)
        if xfer_handle:
            nixl_agent.release_xfer_handle(xfer_handle)
        if NIXL_REMOTE_AGENT_CACHE_MAXSIZE == 0 and remote_name:
            nixl_agent.remove_remote_agent(remote_name)
        if remove_tensor_descs:
            self._remove_tensor_descs(tensors)

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> List["torch.Tensor"]:
        """Receives multiple tensors synchronously.

        同步接收多个 Tensor。
        """
        fetch_request = self.fetch_multiple_tensors(
            obj_id, tensor_transport_metadata, communicator_metadata, target_buffers
        )
        return self.wait_fetch_complete(fetch_request)

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        raise NotImplementedError(
            "NIXL transport does not support send_multiple_tensors, since it is a one-sided transport."
        )

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: List["torch.Tensor"],
    ):
        with self._cache_lock:
            assert isinstance(tensor_transport_meta, NixlTransportMetadata)
            if obj_id not in self._managed_meta_nixl:
                return
            self._managed_meta_nixl.pop(obj_id, None)
            self._remove_tensor_descs(tensors)

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)

    def _get_num_managed_meta_nixl(self) -> int:
        with self._cache_lock:
            return len(self._managed_meta_nixl)

    def _get_meta(self, object_id: str) -> Optional[NixlTransportMetadata]:
        """
        Get the NIXL transport metadata for the given object ID if it exists

        如果存在，获取给定对象 ID 的 NIXL 传输元数据
        """
        with self._cache_lock:
            if object_id in self._managed_meta_nixl:
                return self._managed_meta_nixl[object_id]
            return None

    def _put_meta(self, object_id: str, meta: NixlTransportMetadata):
        """
        Store the NIXL transport metadata for the given object ID

        存储给定对象 ID 的 NIXL 传输元数据
        """
        with self._cache_lock:
            self._managed_meta_nixl[object_id] = meta

    def _remove_tensor_descs(self, tensors: List["torch.Tensor"]):
        """
        Decrements the reference count for each tensor. If the count reaches 0,
        traditionally-registered memory is deregistered from NIXL, while
        pool-managed blocks (reg_desc is None) are returned to the pool.

        递减每个 Tensor 的引用计数。如果计数达到 0，
        传统注册的内存将从 NIXL 注销，而池管理的内存块（reg_desc 为 None）则归还给池。
        """
        with self._cache_lock:
            pool_return_tensors: List["torch.Tensor"] = []
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key not in self._tensor_desc_cache:
                    continue
                tensor_desc = self._tensor_desc_cache[key]
                tensor_desc.metadata_count -= 1
                if tensor_desc.metadata_count == 0:
                    self._tensor_desc_cache.pop(key)
                    if tensor_desc.reg_desc is not None:
                        # Traditional path: deregister NIXL memory.
                        # 传统路径：注销 NIXL 内存。
                        self.get_nixl_agent().deregister_memory(tensor_desc.reg_desc)
                        self._nixl_agent_meta_version += 1
                    else:
                        # Pool path: return block to pool.
                        # 池路径：将内存块归还给池。
                        pool_return_tensors.append(tensor)
            if pool_return_tensors and self._memory_pool is not None:
                self._memory_pool.free_tensors(pool_return_tensors)

    def _add_tensor_descs(self, tensors: List["torch.Tensor"]):
        """
        If this is the first time the tensor is being registered, we register the
        full underlying pytorch storage object with NIXL. Otherwise, we increment the reference count.

        如果这是 Tensor 第一次被注册，我们将完整的底层 PyTorch 存储对象注册到 NIXL。
        否则，我们递增引用计数。
        """
        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key in self._tensor_desc_cache:
                    self._tensor_desc_cache[key].metadata_count += 1
                    continue
                mem_type = "cuda" if tensor.is_cuda else "cpu"
                # the GPU ID of the device the tensor is on.
                # NOTE: we clip this to 0 since the GPU ID is not used for
                # CPU tensors, and get_device returns -1 for CPU tensors.
                # This triggers an error in nixl since it expects an unsigned.
                # Tensor 所在设备的 GPU ID。
                # 注意：我们将其裁剪为 0，因为 GPU ID 不用于 CPU Tensor，
                # 而 get_device 对 CPU Tensor 返回 -1。
                # 这会在 nixl 中触发错误，因为它期望无符号值。
                gpu_id = max(tensor.get_device(), 0)
                # Registering the full underlying pytorch storage object by
                # constructing a memory region with the data pointer, size,
                # GPU ID, and meta info. Doing the equivalent of what nixl
                # does for pytorch tensors internally:
                # https://github.com/ai-dynamo/nixl/blob/dd23ef01bd366aef89fa552f2b042f89a0b45fcb/src/api/python/_api.py#L1034
                # 通过构造包含数据指针、大小、GPU ID 和元信息的内存区域，
                # 注册完整的底层 PyTorch 存储对象。执行与 nixl 内部对
                # PyTorch Tensor 所做的等效操作：
                # https://github.com/ai-dynamo/nixl/blob/dd23ef01bd366aef89fa552f2b042f89a0b45fcb/src/api/python/_api.py#L1034
                try:
                    reg_desc = self.get_nixl_agent().register_memory(
                        [
                            (
                                tensor.untyped_storage().data_ptr(),
                                tensor.untyped_storage().nbytes(),
                                gpu_id,
                                "",
                            )
                        ],
                        mem_type=mem_type,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to register {mem_type} memory with NIXL "
                        f"(size={tensor.untyped_storage().nbytes()} bytes, "
                        f"gpu_id={gpu_id}). "
                        f"Common causes:\n"
                        f"  - Locked memory limit too low: check 'ulimit -l' (should be 'unlimited')\n"
                        f"  - nvidia-peermem kernel module not loaded: check 'lsmod | grep nvidia_peermem'\n"
                        f"  - gdrcopy not installed: check 'lsmod | grep gdrdrv'\n"
                        f"  - IOMMU enabled without passthrough mode\n"
                        f"  - Container cgroup memory restrictions\n"
                        f"Set UCX_LOG_LEVEL=debug for detailed UCX diagnostics."
                    ) from e
                self._tensor_desc_cache[key] = TensorDesc(reg_desc, 1)

    def _tensor_memory_registered(self, t: "torch.Tensor") -> bool:
        """Check if the tensor's memory has been registered with NIXL.

        检查 Tensor 的内存是否已注册到 NIXL。
        """
        entry = self._tensor_desc_cache.get(t.untyped_storage().data_ptr())
        return entry is not None and entry.reg_desc is not None

    def _add_pool_tensor_descs(self, tensors: List["torch.Tensor"]):
        """Add pool-managed tensor entries to the unified _tensor_desc_cache.

        Pool-managed tensors use reg_desc=None since pool memory is registered
        once at pool creation. The metadata_count tracks reference counting
        just like traditional tensors.

        Note: Entries are keyed by the source tensor's storage ``data_ptr()``.
        If PyTorch frees and reallocates that storage address before GC runs,
        a stale cache entry could map to an unrelated tensor. This is the same
        constraint as the traditional (non-pool) path and is mitigated by the
        fact that pool blocks hold a reference to pool memory, not the source
        storage.

        将池管理的 Tensor 条目添加到统一的 _tensor_desc_cache 中。

        池管理的 Tensor 使用 reg_desc=None，因为池内存
        在池创建时一次性注册。metadata_count 跟踪引用计数，
        与传统 Tensor 相同。

        注意：条目以源 Tensor 的存储 ``data_ptr()`` 为键。
        如果 PyTorch 在 GC 运行前释放并重新分配该存储地址，
        过期的缓存条目可能映射到无关的 Tensor。这与传统（非池）
        路径的约束相同，并通过池内存块持有对池内存而非源存储
        的引用来缓解。
        """
        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key in self._tensor_desc_cache:
                    self._tensor_desc_cache[key].metadata_count += 1
                else:
                    self._tensor_desc_cache[key] = TensorDesc(
                        reg_desc=None, metadata_count=1
                    )

    def _allocate_pool_xfer_descs(self, tensors: List["torch.Tensor"]) -> Any:
        """Allocate pool memory for tensors and return NIXL transfer descriptors.

        Handles rollback of newly allocated pool blocks if get_xfer_descs
        fails, without disturbing cached blocks from prior calls.

        为 Tensor 分配池内存并返回 NIXL 传输描述符。

        如果 get_xfer_descs 失败，处理新分配的池内存块的回滚，
        不影响先前调用的缓存块。
        """
        pool = self._memory_pool
        # Remember which storages already have a pool block (cache hits)
        # so we don't free them on rollback.
        # 记住哪些存储已有池内存块（缓存命中），
        # 以便在回滚时不释放它们。
        pre_existing = {
            t.untyped_storage().data_ptr() for t in tensors if pool.has_block(t)
        }
        pool_tensor_views = pool.allocate_for_tensors(tensors)
        try:
            xfer_descs = self._nixl_agent.get_xfer_descs(pool_tensor_views)
        except Exception:
            # Only free newly allocated blocks, not cache hits.
            # 只释放新分配的内存块，不释放缓存命中的块。
            new_tensors = [
                t for t in tensors if t.untyped_storage().data_ptr() not in pre_existing
            ]
            if new_tensors:
                pool.free_tensors(new_tensors)
            raise
        self._add_pool_tensor_descs(tensors)
        return xfer_descs
