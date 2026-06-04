import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportMetadata,
)
from ray.experimental.rdt.util import (
    device_match_transport,
    get_tensor_transport_manager,
)

if TYPE_CHECKING:
    import torch


def __ray_send__(
    self,
    obj_id: str,
    tensor_transport_meta: TensorTransportMetadata,
    communicator_meta: CommunicatorMetadata,
    backend: str,
):
    """Helper function that runs on the src actor to send tensors to the dst actor.
    在源 actor 上运行的辅助函数，用于将 tensor 发送到目标 actor。"""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager._rdt_store
    assert rdt_store.has_object(obj_id), f"obj_id={obj_id} not found in RDT store"

    tensors = rdt_store.get_object(obj_id)

    tensor_transport_manager = get_tensor_transport_manager(backend)
    tensor_transport_manager.send_multiple_tensors(
        tensors,
        tensor_transport_meta,
        communicator_meta,
    )


def validate_tensor_buffers(
    tensor_buffers: List["torch.Tensor"],
    tensor_meta: List[Tuple["torch.Size", "torch.dtype"]],
    device: str,
):
    if len(tensor_buffers) != len(tensor_meta):
        raise ValueError(
            f"Length of tensor_buffers ({len(tensor_buffers)}) does not match length from object metadata ({len(tensor_meta)})."
        )

    def tensor_buffer_mismatch_msg(prop, idx, actual, expected):
        return f"{prop} of tensor_buffer at index {idx} ({actual}) does not match {prop.lower()} from object metadata ({expected})."

    for idx, single_buffer in enumerate(tensor_buffers):
        shape, dtype = tensor_meta[idx]
        if single_buffer.shape != shape:
            raise ValueError(
                tensor_buffer_mismatch_msg("Shape", idx, single_buffer.shape, shape)
            )
        if single_buffer.dtype != dtype:
            raise ValueError(
                tensor_buffer_mismatch_msg("Dtype", idx, single_buffer.dtype, dtype)
            )
        if single_buffer.device.type != device:
            raise ValueError(
                tensor_buffer_mismatch_msg(
                    "Device", idx, single_buffer.device.type, device
                )
            )
        if not single_buffer.is_contiguous():
            raise ValueError(f"Tensor buffer at index {idx} is not contiguous.")


def __ray_recv__(
    self,
    obj_id: str,
    tensor_transport_meta: TensorTransportMetadata,
    communicator_meta: CommunicatorMetadata,
    backend: str,
    target_buffers: Optional[List[Any]] = None,
):
    """Helper function that runs on the dst actor to receive tensors from the src actor.
    在目标 actor 上运行的辅助函数，用于从源 actor 接收 tensor。"""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager.rdt_store
    try:
        tensor_transport_manager = get_tensor_transport_manager(backend)
        if target_buffers:
            # Currently only torch tensors are supported as target buffers. We could make this
            # more generic in the future by adding a pluggable buffer validation function.
            # 目前仅支持 torch tensor 作为目标缓冲区。未来可以通过添加可插拔的缓冲区验证函数来使其更通用。
            validate_tensor_buffers(
                target_buffers,
                tensor_transport_meta.tensor_meta,
                tensor_transport_meta.tensor_device,
            )
        tensors = tensor_transport_manager.recv_multiple_tensors(
            obj_id,
            tensor_transport_meta,
            communicator_meta,
            target_buffers,
        )
        assert len(tensors) == len(tensor_transport_meta.tensor_meta)
        rdt_store.add_object(obj_id, tensors)
    except Exception as e:
        # Store the error as an RDT object if the recv fails, so waiters will raise the error.
        # 如果接收失败，将错误存储为 RDT 对象，以便等待者会抛出该错误。
        rdt_store.add_object(obj_id, e)


def __ray_abort_transport__(
    self, obj_id: str, communicator_meta: CommunicatorMetadata, backend: str
):
    """Helper function that can run on an actor doing a send or recv to abort the transport.
    可在执行发送或接收的 actor 上运行的辅助函数，用于中止传输。"""
    tensor_transport_manager = get_tensor_transport_manager(backend)
    tensor_transport_manager.abort_transport(obj_id, communicator_meta)


def __ray_free__(
    self,
    obj_id: str,
    tensor_transport_backend: str,
    tensor_transport_meta: TensorTransportMetadata,
):
    try:
        from ray._private.worker import global_worker

        tensor_transport_manager = get_tensor_transport_manager(
            tensor_transport_backend
        )
        rdt_manager = global_worker.rdt_manager
        rdt_store = rdt_manager.rdt_store

        if not rdt_store.has_object(obj_id):
            return
        tensors = rdt_store.get_object(obj_id)
        tensor_transport_manager.garbage_collect(obj_id, tensor_transport_meta, tensors)

        rdt_store.pop_object(obj_id)
    except AssertionError:
        # This could fail if this is a retry and it's already been freed.
        # 如果这是重试操作且对象已被释放，此操作可能会失败。
        pass


def __ray_fetch_rdt_object__(self, obj_id: str):
    """Helper function that runs on the src actor to fetch tensors from the RDT store via the object store.
    在源 actor 上运行的辅助函数，通过对象存储从 RDT 存储中获取 tensor。"""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager.rdt_store
    rdt_object = rdt_store.wait_and_get_object(obj_id)
    return rdt_object


@dataclass
class _RDTObject:
    # A list of tensors representing the RDT object.
    # 表示 RDT 对象的 tensor 列表。
    data: List[Any]
    # Whether the RDT object is the primary copy.
    # 该 RDT 对象是否为主副本。
    is_primary: bool
    # If a recv failed, we store the error here.
    # 如果接收失败，将错误存储在此处。
    error: Optional[Exception] = None


class RDTStore:
    """
    This class is thread-safe. The GPU object store is meant to be read and
    written by the following threads:
    1. The main thread, which is executing user code. This thread may get, put,
    and pop objects.
    2. The background _ray_system thread, which executes data transfers. This
    thread may get and put objects.
    3. The background CoreWorker server thread, which executes garbage
    collection callbacks that pop objects that are no longer in use.

    此类是线程安全的。GPU 对象存储旨在被以下线程读写：
    1. 主线程，执行用户代码。此线程可以获取、放入和弹出对象。
    2. 后台 _ray_system 线程，执行数据传输。此线程可以获取和放入对象。
    3. 后台 CoreWorker 服务线程，执行垃圾回收回调，弹出不再使用的对象。
    """

    def __init__(self):
        # A dictionary that maps from an object ID to a queue of tensor lists.
        #
        # Note: Currently, `_rdt_store` is only supported for Ray Actors.
        # 从对象 ID 到 tensor 列表队列的映射字典。
        #
        # 注意：目前 `_rdt_store` 仅支持 Ray Actor。
        self._rdt_store: Dict[str, deque[_RDTObject]] = defaultdict(deque)
        # Mapping from tensor data pointer to the IDs of objects that contain it.
        # 从 tensor 数据指针到包含该 tensor 的对象 ID 的映射。
        self._tensor_to_object_ids: Dict[int, Set[str]] = defaultdict[int, Set[str]](
            set
        )
        # Synchronization for the RDT store.
        # RDT 存储的同步机制。
        self._lock = threading.RLock()
        # Signal when an object becomes present in the object store.
        # 当对象出现在对象存储中时发出信号。
        self._object_present_cv = threading.Condition(self._lock)
        # Signal when an object is freed from the object store.
        # 当对象从对象存储中被释放时发出信号。
        self._object_freed_cv = threading.Condition(self._lock)

    def has_object(self, obj_id: str) -> bool:
        with self._lock:
            existed = obj_id in self._rdt_store
            if existed:
                return len(self._rdt_store[obj_id]) > 0
            return existed

    def has_tensor(self, tensor: Any) -> bool:
        # Method only used for testing.
        # 仅用于测试的方法。
        with self._lock:
            return id(tensor) in self._tensor_to_object_ids

    def get_object(self, obj_id: str) -> Optional[List[Any]]:
        with self._lock:
            if self._rdt_store[obj_id][0].error:
                raise self._rdt_store[obj_id][0].error
            return self._rdt_store[obj_id][0].data

    def add_object(
        self,
        obj_id: str,
        rdt_object: Union[List[Any], Exception],
        is_primary: bool = False,
    ):
        """
        Add an RDT object to the RDT store.

        Args:
            obj_id: The object ID of the RDT object.
            rdt_object: A list of tensors representing the RDT object.
            is_primary: Whether the RDT object is the primary copy.

        将 RDT 对象添加到 RDT 存储中。

        参数:
            obj_id: RDT 对象的对象 ID。
            rdt_object: 表示 RDT 对象的 tensor 列表。
            is_primary: 该 RDT 对象是否为主副本。
        """
        with self._object_present_cv:
            if isinstance(rdt_object, Exception):
                self._rdt_store[obj_id].append(
                    _RDTObject([], is_primary, error=rdt_object)
                )
            else:
                for tensor in rdt_object:
                    self._tensor_to_object_ids[id(tensor)].add(obj_id)
                # Append to the queue instead of overwriting
                # 追加到队列而不是覆盖
                self._rdt_store[obj_id].append(
                    _RDTObject(
                        rdt_object,
                        is_primary,
                    )
                )
            self._object_present_cv.notify_all()

    def add_object_primary(
        self, obj_id: str, tensors: List[Any], tensor_transport: str
    ) -> TensorTransportMetadata:
        with self._object_present_cv:
            # A primary entry may already exist from a prior attempt of the
            # same task (e.g., a task that succeeded and populated the RDT
            # store but whose reply was lost, then got retried). Keep the
            # existing primary — do not re-store — and return metadata
            # derived from it so the metadata matches what `__ray_send__`
            # will actually transmit.
            # 主副本条目可能已从同一任务的先前尝试中存在
            # （例如，一个成功并填充了 RDT 存储但其回复丢失的任务，
            # 之后被重试）。保留现有的主副本——不要重新存储——
            # 并返回从其派生的元数据，以便元数据与 `__ray_send__`
            # 实际传输的内容一致。
            queue = self._rdt_store.get(obj_id)
            if queue:
                tensors_to_describe = queue[0].data
            else:
                self.add_object(obj_id, tensors, is_primary=True)
                tensors_to_describe = tensors

        tensor_transport_manager = get_tensor_transport_manager(tensor_transport)
        tensor_transport_meta = (
            tensor_transport_manager.extract_tensor_transport_metadata(
                obj_id, tensors_to_describe
            )
        )

        if tensor_transport_meta.tensor_meta and not device_match_transport(
            tensor_transport_meta.tensor_device, tensor_transport
        ):
            raise ValueError(
                f"Tensor transport backend {tensor_transport} does not support "
                f"tensor transfer on device {tensor_transport_meta.tensor_device}."
            )

        return tensor_transport_meta

    def is_primary_copy(self, obj_id: str) -> bool:
        with self._lock:
            return self.has_object(obj_id) and self._rdt_store[obj_id][0].is_primary

    def wait_and_get_object(
        self, obj_id: str, timeout: Optional[float] = None
    ) -> List[Any]:
        """Atomically waits for the RDT object to be present in the RDT
        store, then gets it. If the object is not present after the optional
        timeout, raise a TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.

        Returns:
            The tensors in the RDT object.

        原子性地等待 RDT 对象出现在 RDT 存储中，然后获取它。
        如果对象在可选的超时时间后仍未出现，则抛出 TimeoutError。

        参数:
            obj_id: 要等待的对象 ID。
            timeout: 等待对象出现在 RDT 存储中的最大时间（秒）。
                如果未指定，则无限期等待。

        返回:
            RDT 对象中的 tensor。
        """
        with self._lock:
            self._wait_object(obj_id, timeout)
            return self.get_object(obj_id)

    def wait_and_pop_object(
        self, obj_id: str, timeout: Optional[float] = None
    ) -> List[Any]:
        """Atomically waits for the RDT object to be present in the RDT
        store, then pops it.  If the object is not present after the optional
        timeout, raise a TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.

        Returns:
            The RDT object.

        原子性地等待 RDT 对象出现在 RDT 存储中，然后弹出它。
        如果对象在可选的超时时间后仍未出现，则抛出 TimeoutError。

        参数:
            obj_id: 要等待的对象 ID。
            timeout: 等待对象出现在 RDT 存储中的最大时间（秒）。
                如果未指定，则无限期等待。

        返回:
            RDT 对象。
        """
        with self._lock:
            self._wait_object(obj_id, timeout)
            return self.pop_object(obj_id)

    def _wait_object(self, obj_id: str, timeout: Optional[float] = None) -> None:
        """Helper method to wait for the RDT object to be present in the RDT store.
        If the object is not present after the optional timeout, raise a
        TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.

        等待 RDT 对象出现在 RDT 存储中的辅助方法。
        如果对象在可选的超时时间后仍未出现，则抛出 TimeoutError。

        参数:
            obj_id: 要等待的对象 ID。
            timeout: 等待对象出现在 RDT 存储中的最大时间（秒）。
                如果未指定，则无限期等待。
        """
        with self._object_present_cv:
            if not self._object_present_cv.wait_for(
                lambda: self.has_object(obj_id),
                timeout=timeout,
            ):
                raise TimeoutError(
                    f"ObjectRef({obj_id}) not found in RDT object store after {timeout}s, transfer may have failed. Please report this issue on GitHub: https://github.com/ray-project/ray/issues/new/choose"
                )

    def pop_object(self, obj_id: str) -> List[Any]:
        with self._lock:
            queue = self._rdt_store.get(obj_id)
            assert queue is not None, f"obj_id={obj_id} not found in RDT store"
            rdt_object = queue.popleft()
            if len(queue) == 0:
                del self._rdt_store[obj_id]
            if rdt_object.error:
                raise rdt_object.error
            for tensor in rdt_object.data:
                self._tensor_to_object_ids[id(tensor)].remove(obj_id)
                if len(self._tensor_to_object_ids[id(tensor)]) == 0:
                    self._tensor_to_object_ids.pop(id(tensor))
            self._object_freed_cv.notify_all()
            return rdt_object.data

    def wait_tensor_freed(self, tensor: Any, timeout: Optional[float] = None) -> None:
        """
        Wait for the object to be freed from the RDT store.

        等待对象从 RDT 存储中被释放。
        """
        with self._object_freed_cv:
            if not self._object_freed_cv.wait_for(
                lambda: id(tensor) not in self._tensor_to_object_ids,
                timeout=timeout,
            ):
                raise TimeoutError(
                    f"Tensor {tensor} not freed from RDT object store after {timeout}s. The tensor will not be freed until all ObjectRefs containing the tensor have gone out of scope."
                )

    def get_num_objects(self) -> int:
        """
        Return the number of objects in the RDT store.

        返回 RDT 存储中的对象数量。
        """
        with self._lock:
            # Count total objects across all queues
            # 计算所有队列中的对象总数
            return sum(len(queue) for queue in self._rdt_store.values())
