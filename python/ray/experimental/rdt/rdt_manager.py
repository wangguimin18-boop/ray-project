import logging
import threading
import time
import warnings
import weakref
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import ray
from ray._private import ray_constants
from ray._raylet import ObjectRef
from ray.experimental.rdt.tensor_transport_manager import FetchRequest
from ray.util.annotations import PublicAPI


@dataclass
class ObjectStoreFetchRequest(FetchRequest):
    """Pending fetch via the object store. Holds the remote ObjectRef to ray.get on.

    通过对象存储进行的待处理获取操作。持有需要 ray.get 的远程 ObjectRef。

    Args:
        obj_id: The RDT object ID being fetched.
        object_ref: The ObjectRef returned by the __ray_fetch_rdt_object__ remote call.
        tensors: Unused. Tensors are returned directly by ray.get.

    参数:
        obj_id: 正在获取的 RDT 对象 ID。
        object_ref: 由 __ray_fetch_rdt_object__ 远程调用返回的 ObjectRef。
        tensors: 未使用。Tensor 由 ray.get 直接返回。
    """

    object_ref: Optional[ObjectRef] = None
    tensors: Optional[List[Any]] = None


if TYPE_CHECKING:

    from ray.experimental.rdt.rdt_store import (
        RDTStore,
    )
    from ray.experimental.rdt.tensor_transport_manager import (
        CommunicatorMetadata,
        FetchRequest,
        TensorTransportMetadata,
    )

logger = logging.getLogger(__name__)

# RDTMeta is a named tuple containing the source actor, tensor transport
# backend, tensor metadata, and other information that needs to be recorded.
# - The tensor transport backend is the backend used to transport the tensors.
# - The tensor metadata is a list of tuples, each containing the shape and dtype
#   of a tensor in the RDT store.
# RDTMeta 是一个命名元组，包含源 Actor、Tensor 传输后端、Tensor 元数据和其他需要记录的信息。
# - Tensor 传输后端是用于传输 Tensor 的后端。
# - Tensor 元数据是一个元组列表，每个元组包含 RDT 存储中 Tensor 的形状和数据类型。
class RDTMeta(NamedTuple):
    src_actor: "ray.actor.ActorHandle"
    tensor_transport_backend: str
    # This is set when the actual object is created and the metadata makes it back to the owner.
    # For ray.put the owner is the creator so it's immediately set.
    # 此字段在实际对象创建且元数据回传到所有者时设置。
    # 对于 ray.put，所有者就是创建者，因此会立即设置。
    tensor_transport_meta: Optional["TensorTransportMetadata"]
    # sent_dest_actors tracks the set of actor IDs that this object has been sent to.
    # Note that since the set is mutable, it shouldn't be accessed without a lock.
    # sent_dest_actors 跟踪此对象已发送到的 Actor ID 集合。
    # 注意：由于该集合是可变的，不应在没有锁的情况下访问。
    sent_dest_actors: Set[str]
    # sent_to_src_actor_and_others_warned indicates whether the object has already triggered a warning about being sent back to the source actor and other actors simultaneously.
    # sent_to_src_actor_and_others_warned 表示该对象是否已触发过关于同时被发送回源 Actor 和其他 Actor 的警告。
    sent_to_src_actor_and_others_warned: bool
    # If the user set buffers for the object, the object will be fetched directly into the buffers on a ray.get
    # 如果用户为对象设置了缓冲区，对象将在 ray.get 时直接获取到缓冲区中
    target_buffers: Optional[List[weakref.ReferenceType[Any]]]


# This is used to periodically check in on the RDT transfer through the refs from
# __ray_send__ and __ray_recv__ and abort operations in case of failures / timeouts.
# 此类用于定期通过 __ray_send__ 和 __ray_recv__ 返回的引用检查 RDT 传输状态，
# 并在发生失败或超时时中止操作。
class TransferMetadata(NamedTuple):
    src_actor: "ray.actor.ActorHandle"
    dst_actor: "ray.actor.ActorHandle"
    send_ref: Optional[ObjectRef]
    recv_ref: ObjectRef
    communicator_meta: "CommunicatorMetadata"
    backend: str
    obj_id: str
    timeout: float


@PublicAPI(stability="alpha")
def wait_tensor_freed(tensor: Any, timeout: Optional[float] = None):
    """
    Wait for the tensor to be freed.

    等待 Tensor 被释放。

    This function is useful for cases where an actor keeps a reference to a
    tensor after returning the tensor from a task annotated with
    `@ray.method(tensor_transport=...)`. In this case, Ray will store a
    *reference* to the tensor, so any in-place modifications made by the actor
    that returned the tensor could be seen by other actors. See
    :ref:`Ray Direct Transport (RDT) <direct-transport>` for more details.

    此函数适用于 Actor 在从带有 `@ray.method(tensor_transport=...)` 注解的任务返回
    Tensor 后仍保留对 Tensor 引用的场景。在这种情况下，Ray 会存储对该 Tensor 的
    *引用*，因此返回 Tensor 的 Actor 所做的任何原地修改都可能被其他 Actor 看到。
    详见 :ref:`Ray Direct Transport (RDT) <direct-transport>`。

    Call this function for RDT objects to ensure that all corresponding
    `ray.ObjectRefs` have gone out of scope and therefore the tensor is safe to
    write to again.

    对 RDT 对象调用此函数，以确保所有对应的 `ray.ObjectRef` 已超出作用域，
    从而确保 Tensor 可以安全地再次写入。

    Args:
        tensor: The tensor to wait to be freed. This should be a tensor that was
            previously returned by a task annotated with
            `@ray.method(tensor_transport=...)` or stored via
            `ray.put(_tensor_transport="...")`.
        timeout: The timeout in seconds to wait for all references to the tensor
            to go out of scope. Set to None to wait indefinitely. Note that if
            None is used, this function could hang if the `ray.ObjectRefs` that
            refer to this tensor never go out of scope.

    参数:
        tensor: 要等待释放的 Tensor。应是通过带有
            `@ray.method(tensor_transport=...)` 注解的任务返回的 Tensor，
            或通过 `ray.put(_tensor_transport="...")` 存储的 Tensor。
        timeout: 等待所有对 Tensor 的引用超出作用域的超时时间（秒）。
            设置为 None 表示无限等待。注意：如果使用 None，
            当引用此 Tensor 的 `ray.ObjectRef` 永不超出作用域时，
            此函数可能会挂起。
    """
    rdt_manager = ray.worker.global_worker.rdt_manager
    rdt_manager.rdt_store.wait_tensor_freed(tensor, timeout)


@PublicAPI(stability="alpha")
def set_target_for_ref(ref: ObjectRef, target: List[Any]):
    """
    Set target buffers for an RDT ObjectRef to fetch tensors into when `ray.get` is called.

    为 RDT ObjectRef 设置目标缓冲区，以便在调用 `ray.get` 时将 Tensor 获取到缓冲区中。

    This is only supported by some transports (e.g., NIXL). If the transport
    does not support this feature, an exception will be raised during ray.get.

    此功能仅被某些传输方式支持（例如 NIXL）。如果传输方式不支持此功能，
    在 ray.get 时将抛出异常。

    Before receiving, Ray validates that the provided target buffers match the metadata
    of the tensors in the object (e.g., shape, dtype, device). If validation fails,
    a `ValueError` is raised. We recommend sending over lists of tensors and passing a list
    of the same length here because the serialization order from the sender-side must match
    the order of the target tensors here.

    在接收之前，Ray 会验证提供的目标缓冲区是否与对象中 Tensor 的元数据匹配
    （例如形状、数据类型、设备）。如果验证失败，将抛出 `ValueError`。
    我们建议发送 Tensor 列表并在此处传入相同长度的列表，
    因为发送端的序列化顺序必须与此处目标 Tensor 的顺序匹配。

    Args:
        ref: The ObjectRef to set the target buffers for. The ref must be for an RDT object.
        target: A list of tensors to be used as the target buffers to receive into.

    参数:
        ref: 要设置目标缓冲区的 ObjectRef。该引用必须是 RDT 对象的引用。
        target: 用作目标缓冲区的 Tensor 列表，用于接收数据。
    """
    rdt_manager = ray.worker.global_worker.rdt_manager
    rdt_manager.set_target_buffers_for_ref(ref, target)


class RDTManager:
    def __init__(self):
        # This lock protects _managed_rdt_metadata, _queued_transfers, and _queued_frees since
        # they can be accessed from the user's python thread or the CoreWorker's main io service thread.
        # 此锁保护 _managed_rdt_metadata、_queued_transfers 和 _queued_frees，
        # 因为它们可以从用户的 Python 线程或 CoreWorker 的主 IO 服务线程访问。
        self._lock = threading.Lock()

        # A dictionary that maps from owned object's ID to RDTMeta.
        # This dictionary is hosted on the "driver" process of the actors that
        # store and send/receive RDT objects.
        # 一个将所拥有对象的 ID 映射到 RDTMeta 的字典。
        # 此字典托管在存储和发送/接收 RDT 对象的 Actor 的"驱动"进程上。
        self._managed_rdt_metadata: Dict[str, RDTMeta] = {}
        # Condition variable to wait for the tensor transport meta to be set.
        # 用于等待 Tensor 传输元数据被设置的条件变量。
        self._tensor_transport_meta_cv = threading.Condition(self._lock)

        # A dictionary that maps from an object id to a list of actors
        # that are queued to receive the object.
        # 一个将对象 ID 映射到排队等待接收该对象的 Actor 列表的字典。
        self._queued_transfers: Dict[str, List["ray.actor.ActorHandle"]] = defaultdict(
            list
        )
        # A set of object ids that are queued to be freed. This is used when the object is freed
        # before the owner knows it's created (the tensor transport metadata is not available yet).
        # 排队等待释放的对象 ID 集合。此集合用于在所有者知晓对象已创建之前
        # （Tensor 传输元数据尚不可用时）对象被释放的情况。
        self._queued_frees: Set[str] = set()

        # This lock makes sure the _rdt_store and _monitor_failures_thread are only created once.
        # 此锁确保 _rdt_store 和 _monitor_failures_thread 只被创建一次。
        self._init_lock = threading.Lock()

        # Per-actor local storage for RDT objects. We create the RDT store
        # lazily, if a user specifies a non-default tensor_transport, to avoid
        # circular import and because it imports third-party dependencies like
        # PyTorch.
        # 每个 Actor 的 RDT 对象本地存储。我们延迟创建 RDT 存储，
        # 仅在用户指定非默认 tensor_transport 时才创建，
        # 以避免循环导入，并因为它会导入第三方依赖如 PyTorch。
        self._rdt_store: Optional["RDTStore"] = None

        # Thread safe queue of transport refs that the monitor thread needs to start monitoring
        # 线程安全的传输引用队列，监控线程需要开始监控这些引用
        self._unmonitored_transfers: Queue[TransferMetadata] = Queue()
        # Background thread to poll on the transfer operation.
        # 用于轮询传输操作的后台线程。
        self._monitor_failures_thread = None
        # Event to signal the monitor_failures thread to shutdown
        # 用于通知 monitor_failures 线程关闭的事件
        self._monitor_failures_shutdown_event = threading.Event()

        # If the actor isn't in the dict, the task to launch the custom transport registration task hasn't been submitted yet.
        # If the value is an object ref, we have to wait for the registration task to complete.
        # If the value is True, the actor has registered any custom transports.
        # The value should never be False.
        # TODO: This is a short-term solution. In the future, we'll do registration with actor initialization
        # to make actor restarts and submitting from another worker work.
        # 如果 Actor 不在字典中，则表示启动自定义传输注册任务的请求尚未提交。
        # 如果值是 ObjectRef，则需要等待注册任务完成。
        # 如果值是 True，则表示该 Actor 已注册所有自定义传输。
        # 值不应为 False。
        # TODO: 这是一个短期解决方案。未来我们将通过 Actor 初始化进行注册，
        # 以使 Actor 重启和从其他 Worker 提交都能正常工作。
        self.actor_id_to_transports_registered: Dict[str, Union[ObjectRef, bool]] = {}

    def register_custom_transports_on_actor(self, actor: "ray.actor.ActorHandle"):
        from ray.experimental.rdt.util import (
            register_custom_tensor_transports_on_actor,
        )

        ref = register_custom_tensor_transports_on_actor(actor)
        # ref is None if there are no custom transports registered.
        # 如果没有注册自定义传输，ref 为 None。
        self.actor_id_to_transports_registered[actor._actor_id] = (
            True if ref is None else ref
        )

    def wait_until_custom_transports_registered(self, actor: "ray.actor.ActorHandle"):
        actor_id = actor._actor_id
        if actor_id not in self.actor_id_to_transports_registered:
            self.register_custom_transports_on_actor(actor)

        if self.actor_id_to_transports_registered[actor_id] is not True:
            ray.get(self.actor_id_to_transports_registered[actor_id])
            self.actor_id_to_transports_registered[actor_id] = True

    @property
    def rdt_store(self) -> "ray.experimental.RDTStore":
        with self._init_lock:
            if self._rdt_store is None:
                from ray.experimental.rdt.rdt_store import (
                    RDTStore,
                )

                self._rdt_store = RDTStore()
            return self._rdt_store

    def shutdown(self):
        """
        Interrupt and join the _monitor_failures_thread

        中断并等待 _monitor_failures_thread 线程结束
        """
        with self._init_lock:
            if self._monitor_failures_thread:
                self._monitor_failures_shutdown_event.set()
                self._monitor_failures_thread.join()
                self._monitor_failures_shutdown_event.clear()
                self._monitor_failures_thread = None

    def start_monitor_thread_if_needed(self):
        with self._init_lock:
            # To make sure _monitor_failures_thread is started only once
            # 确保 _monitor_failures_thread 只启动一次
            if self._monitor_failures_thread is None:
                self._monitor_failures_thread = threading.Thread(
                    target=self._monitor_failures, daemon=True
                )
                self._monitor_failures_thread.start()

    def is_managed_object(self, obj_id: str) -> bool:
        """
        Check if the RDT object is owned or borrowed by this process.

        检查 RDT 对象是否由本进程拥有或借用。
        """
        with self._lock:
            return obj_id in self._managed_rdt_metadata

    def set_rdt_metadata(self, obj_id: str, rdt_meta: RDTMeta):
        with self._lock:
            self._managed_rdt_metadata[obj_id] = rdt_meta

    def get_rdt_metadata(self, obj_id: str) -> Optional[RDTMeta]:
        with self._lock:
            return self._managed_rdt_metadata.get(obj_id, None)

    def wait_for_tensor_transport_metadata(
        self, obj_id: str, timeout: float
    ) -> Optional["TensorTransportMetadata"]:
        with self._tensor_transport_meta_cv:
            if self._tensor_transport_meta_cv.wait_for(
                lambda: self._managed_rdt_metadata[obj_id].tensor_transport_meta
                is not None,
                timeout=timeout,
            ):
                return self._managed_rdt_metadata[obj_id].tensor_transport_meta
            else:
                return None

    def _monitor_failures(self):
        """
        Monitor the refs from send and recv tasks and abort the transfers
        if they error out or timeout to prevent hanging.

        监控发送和接收任务的引用，如果出错或超时则中止传输以防止挂起。
        """
        not_done = []
        done = []
        ref_info_map = {}
        while not self._monitor_failures_shutdown_event.is_set():
            while not self._unmonitored_transfers.empty():
                ref_info = self._unmonitored_transfers.get()
                if ref_info.send_ref:
                    not_done.append(ref_info.send_ref)
                    ref_info_map[ref_info.send_ref.hex()] = ref_info
                not_done.append(ref_info.recv_ref)
                ref_info_map[ref_info.recv_ref.hex()] = ref_info
            if len(not_done) > 0:
                done, not_done = ray.wait(not_done, num_returns=1, timeout=1)
            if len(done) > 0:
                try:
                    ray.get(done[0])
                    ref_info_map.pop(done[0].hex(), None)
                except Exception as e:
                    self._abort_transport(done[0], ref_info_map, e)

            while len(not_done) > 0:
                if not_done[0].hex() not in ref_info_map:
                    # The associated transfer was already aborted.
                        # 相关的传输已被中止。
                    not_done.pop(0)
                elif ref_info_map[not_done[0].hex()].timeout < time.time():
                    self._abort_transport(
                        not_done[0],
                        ref_info_map,
                        TimeoutError(
                            f"RDT transfer failed after {ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS}s. "
                            "You can increase the timeout by setting RAY_rdt_fetch_fail_timeout_milliseconds"
                        ),
                    )
                else:
                    # wait returns lists in the same order they were passed in, so if
                    # the timeout of first hasn't been reached, neither have the others.
                    # wait 返回的列表顺序与传入顺序相同，因此如果第一个的超时
                    # 时间尚未到达，其余的也不会到达。
                    break
            if len(not_done) == 0:
                # If we emptied out _unmonitored_transfers on this iteration, wait for a bit.
                # 如果在此次迭代中清空了 _unmonitored_transfers，等待一会儿。
                self._monitor_failures_shutdown_event.wait(1)

    def _abort_transport(
        self,
        failed_ref: ObjectRef,
        ref_info_map: Dict[str, TransferMetadata],
        exception: Exception,
    ):
        """
        Cleans up the ref_info_map, kill the src and dst actors, and destroy the
        collective group if necessary.

        清理 ref_info_map，终止源和目标 Actor，并在必要时销毁集合通信组。
        """
        from ray.experimental.collective import destroy_collective_group
        from ray.experimental.rdt.collective_tensor_transport import (
            CollectiveCommunicatorMetadata,
        )
        from ray.experimental.rdt.rdt_store import (
            __ray_abort_transport__,
        )
        from ray.experimental.rdt.util import (
            get_tensor_transport_manager,
        )

        ref_info = ref_info_map.pop(failed_ref.hex(), None)
        if ref_info is None:
            return

        if ref_info.send_ref:
            ref_info_map.pop(ref_info.send_ref.hex(), None)
        ref_info_map.pop(ref_info.recv_ref.hex(), None)

        tensor_transport_manager = get_tensor_transport_manager(ref_info.backend)
        if tensor_transport_manager.can_abort_transport():
            if not tensor_transport_manager.__class__.is_one_sided():
                # This is dead code until we implement a NCCL abort since NIXL
                # is the only abortable transport for now and is one-sided.
                # 此段代码目前是死代码，因为我们尚未实现 NCCL 中止功能，
                # NIXL 是目前唯一支持中止的传输方式且是单向的。
                ref_info.src_actor.__ray_call__.options(
                    concurrency_group="_ray_system_error"
                ).remote(
                    __ray_abort_transport__,
                    ref_info.obj_id,
                    ref_info.communicator_meta,
                    ref_info.backend,
                )
            ref_info.dst_actor.__ray_call__.options(
                concurrency_group="_ray_system_error"
            ).remote(
                __ray_abort_transport__,
                ref_info.obj_id,
                ref_info.communicator_meta,
                ref_info.backend,
            )
            logger.info(
                "RDT transfer with src actor %s and dst actor %s failed due to %s.",
                ref_info.src_actor,
                ref_info.dst_actor,
                exception,
            )
        else:
            # TODO(#51276): Kill all actors in the collective group when we support more collective operations
            # TODO(#51276): 当我们支持更多集合操作时，终止集合组中的所有 Actor
            ray.kill(ref_info.src_actor)
            ray.kill(ref_info.dst_actor)
            logger.error(
                "RDT transfer with src actor %s and dst actor %s failed. Killing the actors. "
                "Transfer failed with exception: %s",
                ref_info.src_actor,
                ref_info.dst_actor,
                exception,
            )

        # isinstance does an implicit cast and makes communicator_name inaccessible
        # so we have to get communicator_name before the cast.
        # isinstance 会进行隐式类型转换，使 communicator_name 不可访问，
        # 因此我们必须在类型转换之前获取 communicator_name。
        if isinstance(ref_info.communicator_meta, CollectiveCommunicatorMetadata):
            try:
                collective_group_name = ref_info.communicator_meta.communicator_name
                destroy_collective_group(collective_group_name)
                logger.error(
                    "Destroyed collective group %s due to a hanging/failed RDT transfer",
                    collective_group_name,
                )
            except ValueError:
                # Collective group was already destroyed
                # 集合通信组已被销毁
                pass

    def add_rdt_ref(
        self,
        obj_ref: ObjectRef,
        src_actor: "ray.actor.ActorHandle",
        tensor_transport: str,
        tensor_transport_meta: Optional["TensorTransportMetadata"] = None,
    ):
        """Add an RDT object reference to the RDT manager. This should be
        called whenever the current process calls a task that is annotated with
        `@ray.method(tensor_transport=...)`.

        将 RDT 对象引用添加到 RDT 管理器。每当当前进程调用带有
        `@ray.method(tensor_transport=...)` 注解的任务时，应调用此方法。

        Args:
            obj_ref: The ObjectRef of the task output.
            src_actor: The actor that executes the task and that creates the RDT object.
            tensor_transport: The tensor transport protocol to use for the RDT object.
            tensor_transport_meta: The tensor transport metadata that is pre-computed.
                This is known at ref creation time if the object is created through ray.put.

        参数:
            obj_ref: 任务输出的 ObjectRef。
            src_actor: 执行任务并创建 RDT 对象的 Actor。
            tensor_transport: 用于 RDT 对象的 Tensor 传输协议。
            tensor_transport_meta: 预计算的 Tensor 传输元数据。
                如果对象是通过 ray.put 创建的，此值在引用创建时就已知。
        """
        self.set_rdt_metadata(
            obj_ref.hex(),
            RDTMeta(
                src_actor=src_actor,
                tensor_transport_backend=tensor_transport,
                tensor_transport_meta=tensor_transport_meta,  # None if not from ray.put / 如果不是来自 ray.put 则为 None
                sent_dest_actors=set(),
                sent_to_src_actor_and_others_warned=False,
                target_buffers=None,
            ),
        )

    def set_tensor_transport_metadata_and_trigger_queued_operations(
        self, obj_id: str, tensor_transport_meta: "TensorTransportMetadata"
    ):
        """
        Sets the tensor transport metadata for an object and triggers any queued
        up transfers or frees for that object.

        为对象设置 Tensor 传输元数据，并触发该对象所有排队的传输或释放操作。
        """
        dst_actors = None
        free_object = False
        with self._tensor_transport_meta_cv:
            self._managed_rdt_metadata[obj_id] = self._managed_rdt_metadata[
                obj_id
            ]._replace(tensor_transport_meta=tensor_transport_meta)
            dst_actors = self._queued_transfers.pop(obj_id, None)
            free_object = obj_id in self._queued_frees
            if free_object:
                self._queued_frees.remove(obj_id)
                # There shouldn't be any transfers queued if the free was queued,
                # since we clear the queued transfers when queueing the free.
                # 如果释放操作已排队，则不应有任何传输操作排队，
                # 因为在排队释放时会清除排队的传输操作。
                assert dst_actors is None
            self._tensor_transport_meta_cv.notify_all()

        if free_object:
            self.free_object_primary_copy(obj_id)
        if dst_actors:
            for dst_actor in dst_actors:
                # Trigger the transfer now that the metadata is available.
                # 元数据已可用，现在触发传输。
                self.trigger_out_of_band_tensor_transfer(dst_actor, obj_id)

    def set_target_buffers_for_ref(self, ref: ObjectRef, target_buffers: List[Any]):
        with self._lock:
            if ref.hex() not in self._managed_rdt_metadata:
                raise ValueError(f"Ref {ref} is not an RDT object.")

            self._managed_rdt_metadata[ref.hex()] = self._managed_rdt_metadata[
                ref.hex()
            ]._replace(
                target_buffers=[
                    weakref.ref(target_buffer) for target_buffer in target_buffers
                ]
            )

    def _trigger_fetch(
        self,
        obj_id: str,
        use_object_store: bool,
    ) -> FetchRequest:
        """
        Start fetching an RDT object.

        开始获取一个 RDT 对象。

        If the specified transport supports async fetches, this will trigger the
        fetch without blocking. Note that this always triggers a fetch, even if
        the object is already in the store.

        如果指定的传输方式支持异步获取，此方法将在不阻塞的情况下触发获取。
        注意：此方法始终会触发获取，即使对象已在存储中。

        Args:
            obj_id: The object ID of the RDT object.
            use_object_store: Whether to fetch through the object store or through
                the designated one-sided tensor transport.

        Returns:
            A FetchRequest. Wait on the FetchRequest to get the tensors.

        参数:
            obj_id: RDT 对象的对象 ID。
            use_object_store: 是否通过对象存储获取，还是通过指定的单向 Tensor 传输获取。

        返回:
            一个 FetchRequest。等待 FetchRequest 以获取 Tensor。
        """
        from ray.experimental.rdt.rdt_store import (
            __ray_fetch_rdt_object__,
        )
        from ray.experimental.rdt.util import (
            get_tensor_transport_manager,
            is_one_sided_transport,
        )

        rdt_meta = self.get_rdt_metadata(obj_id)
        assert rdt_meta is not None

        if use_object_store:
            if rdt_meta.target_buffers:
                logger.warning(
                    "Target buffers are not supported for use_object_store=True. Ignoring the target buffers."
                )

            src_actor = rdt_meta.src_actor
            object_ref = src_actor.__ray_call__.options(
                concurrency_group="_ray_system"
            ).remote(__ray_fetch_rdt_object__, obj_id)
            return ObjectStoreFetchRequest(
                obj_id=obj_id, object_ref=object_ref, tensors=[]
            )
        else:
            tensor_transport = rdt_meta.tensor_transport_backend
            if not is_one_sided_transport(tensor_transport):
                raise ValueError(
                    f"ray.get is not allowed on RDT objects using the two-sided transport {tensor_transport}. "
                    "Either use a one-sided RDT transport or pass _use_object_store=True to ray.get to fetch the object through the object store instead."
                )
            tensor_transport_manager = get_tensor_transport_manager(tensor_transport)
            communicator_meta = tensor_transport_manager.get_communicator_metadata(
                None, None, tensor_transport
            )

            tensor_transport_meta = rdt_meta.tensor_transport_meta
            if tensor_transport_meta is None:
                # We can't fetch the object until we know the creator has actually created the object.
                # 在知道创建者已实际创建对象之前，我们无法获取该对象。
                timeout = ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS
                tensor_transport_meta = self.wait_for_tensor_transport_metadata(
                    obj_id, timeout
                )
                if tensor_transport_meta is None:
                    raise TimeoutError(
                        f"Timed out after {timeout}s waiting for object {obj_id} to be created while trying to get the object. "
                        "You can increase the timeout by setting RAY_rdt_fetch_fail_timeout_milliseconds."
                    )

            target_buffers = None
            if rdt_meta.target_buffers:
                # Try to get the target buffers from the weak references. If any of the
                # target buffers are not alive, we just won't use the target buffers.
                # 尝试从弱引用中获取目标缓冲区。如果任何目标缓冲区不再存活，
                # 我们将不使用目标缓冲区。
                target_buffers = []
                for target_buffer in rdt_meta.target_buffers:
                    buffer = target_buffer()
                    if buffer is None:
                        target_buffers = None
                        break
                    else:
                        target_buffers.append(buffer)

            if target_buffers is not None:
                from ray.experimental.rdt.rdt_store import validate_tensor_buffers

                device = tensor_transport_meta.tensor_device
                tensor_meta = tensor_transport_meta.tensor_meta
                validate_tensor_buffers(target_buffers, tensor_meta, device)

            return tensor_transport_manager.fetch_multiple_tensors(
                obj_id,
                tensor_transport_meta,
                communicator_meta,
                target_buffers,
            )

    def _wait_fetch(
        self, obj_id: str, fetch_request: FetchRequest, timeout: float = -1
    ) -> List[Any]:
        """
        Waits for a previously triggered fetch to complete and returns the tensors.

        等待先前触发的获取操作完成，并返回 Tensor。

        Args:
            obj_id: The object ID of the RDT object.
            fetch_request: An ObjectStoreFetchRequest representing an object
                transferred via Ray's object store or a FetchRequest
                representing an object transferred via a tensor transport.
            timeout: Maximum time in seconds to wait. -1 means wait indefinitely.
                0 means return immediately if not ready.

        Returns:
            The list of tensors fetched.

        参数:
            obj_id: RDT 对象的对象 ID。
            fetch_request: 一个 ObjectStoreFetchRequest，表示通过 Ray 对象存储传输的对象，
                或一个 FetchRequest，表示通过 Tensor 传输的对象。
            timeout: 最大等待时间（秒）。-1 表示无限等待。
                0 表示如果未就绪则立即返回。

        返回:
            获取的 Tensor 列表。
        """
        if isinstance(fetch_request, ObjectStoreFetchRequest):
            return ray.get(fetch_request.object_ref, timeout=timeout)
        else:
            from ray.experimental.rdt.util import get_tensor_transport_manager

            rdt_meta = self.get_rdt_metadata(obj_id)
            tensor_transport_manager = get_tensor_transport_manager(
                rdt_meta.tensor_transport_backend
            )
            return tensor_transport_manager.wait_fetch_complete(
                fetch_request, timeout=timeout
            )

    def queue_or_trigger_out_of_band_tensor_transfer(
        self, dst_actor: "ray.actor.ActorHandle", task_args: Tuple[Any, ...]
    ):
        """
        Triggers the transfer if the tensor metadata is available for the object. If it's
        not available, the transfer is queued up until the metadata is available.

        如果对象的 Tensor 元数据可用，则触发传输。如果不可用，
        则将传输排队等待元数据可用。
        """
        rdt_object_ids: Set[str] = set()
        for arg in task_args:
            # If an ObjectRef is managed, it means the actual value is a list of tensors stored
            # on a remote actor. Therefore, this function will trigger a tensor communication
            # operation between the sender and receiver actors.
            # 如果 ObjectRef 是被管理的，意味着其实际值是存储在远程 Actor 上的 Tensor 列表。
            # 因此，此函数将触发发送方和接收方 Actor 之间的 Tensor 通信操作。
            if not isinstance(arg, ObjectRef):
                continue
            obj_id = arg.hex()
            if self.is_managed_object(obj_id):
                rdt_object_ids.add(obj_id)
        if rdt_object_ids:
            self.wait_until_custom_transports_registered(dst_actor)
            for obj_id in rdt_object_ids:
                # Atomically gets the tensor transport metadata for an object and queues up a transfer
                # if the tensor transport metadata is not available.
                # 原子性地获取对象的 Tensor 传输元数据，如果元数据不可用则排队等待传输。
                with self._lock:
                    tensor_transport_meta = self._managed_rdt_metadata[
                        obj_id
                    ].tensor_transport_meta
                    if tensor_transport_meta is None:
                        self._queued_transfers[obj_id].append(dst_actor)
                if tensor_transport_meta is not None:
                    self.trigger_out_of_band_tensor_transfer(dst_actor, obj_id)

    def trigger_out_of_band_tensor_transfer(
        self, dst_actor: "ray.actor.ActorHandle", obj_id: str
    ):
        """
        Triggers tensor communication operations between actors. When a managed ObjectRef is passed
        to another actor task, CPU data will still be passed through the object store, but the in-actor
        tensors will be passed out-of-band.

        触发 Actor 之间的 Tensor 通信操作。当被管理的 ObjectRef 传递给另一个 Actor 任务时，
        CPU 数据仍通过对象存储传递，但 Actor 内的 Tensor 将通过带外方式传递。

        This function triggers the out-of-band tensor transfer by submitting Ray actor
        tasks `__ray_send__` to the sender actor and `__ray_recv__` to the receiver actor to initiate
        tensor communication using protocols like NCCL or GLOO.

        此函数通过向发送方 Actor 提交 `__ray_send__` Ray Actor 任务和向接收方 Actor 提交
        `__ray_recv__` Ray Actor 任务来触发带外 Tensor 传输，以使用 NCCL 或 GLOO 等协议
        启动 Tensor 通信。

        Before the receiver actor executes the actor task, the deserializer combines the
        CPU data with the tensors from the sender actor to reconstruct the original task output
        generated by the sender actor.

        在接收方 Actor 执行 Actor 任务之前，反序列化器将 CPU 数据与来自发送方 Actor 的
        Tensor 组合，以重建发送方 Actor 生成的原始任务输出。

        Args:
            dst_actor: The target actor to receive tensors
            obj_id: ID of the object to send to the dst_actor.

        Returns:
            None

        参数:
            dst_actor: 接收 Tensor 的目标 Actor。
            obj_id: 要发送到 dst_actor 的对象 ID。

        返回:
            None
        """
        from ray.experimental.rdt.rdt_store import (
            __ray_recv__,
            __ray_send__,
        )
        from ray.experimental.rdt.util import (
            get_tensor_transport_manager,
        )

        with self._lock:
            # Since sent_dest_actors is mutable, this whole block needs to be protected.
            # 由于 sent_dest_actors 是可变的，整个代码块需要被保护。
            rdt_meta = self._managed_rdt_metadata[obj_id]
            src_actor = rdt_meta.src_actor
            tensor_transport_meta = rdt_meta.tensor_transport_meta

            # Update the set of destination actors for this object
            # The set inside NamedTuple is mutable, so we can modify it directly
            # 更新此对象的目标 Actor 集合
            # NamedTuple 内的集合是可变的，因此我们可以直接修改
            rdt_meta.sent_dest_actors.add(dst_actor._actor_id)
            # Check if a warning should be triggered for this object:
            # 1. object has not triggered a warning yet.
            # 2. object is sent back to its source actor.
            # 3. object is also sent to at least one other actor
            # 检查是否应为此对象触发警告：
            # 1. 对象尚未触发过警告。
            # 2. 对象被发送回其源 Actor。
            # 3. 对象还被发送到至少一个其他 Actor
            if (
                not rdt_meta.sent_to_src_actor_and_others_warned
                and src_actor._actor_id in rdt_meta.sent_dest_actors
                and len(rdt_meta.sent_dest_actors) > 1
            ):
                warnings.warn(
                    f"RDT ObjectRef({obj_id}) is being passed back to the actor that created it {src_actor}. "
                    "Note that RDT objects are mutable. If the tensor is modified, Ray's internal copy will "
                    "also be updated, and subsequent passes to other actors will receive the updated version "
                    "instead of the original.",
                    UserWarning,
                )
                # Mark the object as warned so that we don't warn again for this object.
                # 将对象标记为已警告，以免对此对象再次发出警告。
                self._managed_rdt_metadata[obj_id] = rdt_meta._replace(
                    sent_to_src_actor_and_others_warned=True
                )

            if src_actor._actor_id == dst_actor._actor_id:
                # If the source and destination actors are the same, the tensors can
                # be transferred intra-process, so we skip the out-of-band tensor
                # transfer.
                # 如果源和目标 Actor 相同，Tensor 可以在进程内传输，
                # 因此我们跳过带外 Tensor 传输。
                return

            tensor_transport_manager = get_tensor_transport_manager(
                rdt_meta.tensor_transport_backend
            )
            communicator_meta = tensor_transport_manager.get_communicator_metadata(
                src_actor,
                dst_actor,
                rdt_meta.tensor_transport_backend,
            )

            send_ref = None
            if not tensor_transport_manager.__class__.is_one_sided():
                # Send tensors stored in the `src_actor`'s GPU object store to the
                # destination rank `dst_rank`.
                # NOTE: We put this task on the background thread to avoid tasks
                # executing on the main thread blocking the data transfer.
                # 将存储在 `src_actor` 的 GPU 对象存储中的 Tensor 发送到
                # 目标 Actor `dst_rank`。
                # 注意：我们将此任务放在后台线程上，以避免在主线程上执行的任务阻塞数据传输。
                send_ref = src_actor.__ray_call__.options(
                    concurrency_group="_ray_system"
                ).remote(
                    __ray_send__,
                    obj_id,
                    tensor_transport_meta,
                    communicator_meta,
                    rdt_meta.tensor_transport_backend,
                )

            # Receive tensors from the source rank and store them in the
            # `dst_actor`'s GPU object store.
            # NOTE: Putting this task on the background thread is technically only
            # needed for the sender task, but we put the receiver task on the same
            # background thread to ensure that all communication operations are
            # executed in a global order.
            # 从源 Actor 接收 Tensor 并存储在 `dst_actor` 的 GPU 对象存储中。
            # 注意：将此任务放在后台线程上在技术上仅对发送方任务有必要，
            # 但我们将接收方任务也放在同一后台线程上，以确保所有通信操作
            # 以全局顺序执行。
            recv_ref = dst_actor.__ray_call__.options(
                concurrency_group="_ray_system"
            ).remote(
                __ray_recv__,
                obj_id,
                tensor_transport_meta,
                communicator_meta,
                rdt_meta.tensor_transport_backend,
            )

        self._unmonitored_transfers.put(
            TransferMetadata(
                src_actor=src_actor,
                dst_actor=dst_actor,
                send_ref=send_ref,
                recv_ref=recv_ref,
                communicator_meta=communicator_meta,
                backend=rdt_meta.tensor_transport_backend,
                obj_id=obj_id,
                timeout=time.time() + ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS,
            )
        )
        self.start_monitor_thread_if_needed()

    def get_rdt_objects(
        self,
        object_ids: List[str],
    ) -> Dict[str, List[Any]]:
        """
        Get RDT objects that have already been transferred (e.g. via __ray_recv__).

        获取已传输的 RDT 对象（例如通过 __ray_recv__）。

        This is used in the task argument deserialization path where the
        out-of-band tensor transfer has already been triggered by the caller.
        It only waits on the local RDT store for the tensors to arrive.

        此方法用于任务参数反序列化路径，此时带外 Tensor 传输已由调用方触发。
        它只在本地 RDT 存储上等待 Tensor 到达。

        Args:
            object_ids: The object IDs of the RDT objects.

        Returns:
            A dict mapping object ID to the RDT object (list of tensors).

        参数:
            object_ids: RDT 对象的对象 ID 列表。

        返回:
            一个将对象 ID 映射到 RDT 对象（Tensor 列表）的字典。
        """
        rdt_store = self.rdt_store
        result: Dict[str, List[Any]] = {}
        for object_id in object_ids:
            pop_object = not rdt_store.is_primary_copy(object_id)
            if pop_object:
                result[object_id] = rdt_store.wait_and_pop_object(
                    object_id, timeout=ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS
                )
            else:
                result[object_id] = rdt_store.wait_and_get_object(
                    object_id, timeout=ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS
                )
        return result

    def fetch_and_get_rdt_objects(
        self,
        object_ids: List[str],
        timeout: Optional[float] = None,
        use_object_store: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Fetch and get RDT objects for a list of object IDs, pipelining async fetches.

        为一组对象 ID 获取并得到 RDT 对象，流水线化异步获取操作。

        This is used in the ray.get codepath where the caller initiates the
        tensor fetch. For one-sided transports (e.g. NIXL), all transfers are
        triggered first before waiting, eliminating serial transfer latency.

        此方法用于 ray.get 代码路径，由调用方发起 Tensor 获取。对于单向传输方式
        （例如 NIXL），所有传输会先被触发再等待，从而消除串行传输延迟。

        Args:
            object_ids: The object IDs of the RDT objects.
            timeout: The user-specified timeout from ray.get, or None for no
                user timeout. The actual deadline is the minimum of this and
                RDT_FETCH_FAIL_TIMEOUT_SECONDS.
            use_object_store: Whether to fetch through the object store or through
                the designated tensor transport.

        Returns:
            A dict mapping object ID to the RDT object (list of tensors).

        Raises:
            GetTimeoutError: If the user-specified timeout is exceeded.
            ObjectFetchTimedOutError: If RDT_FETCH_FAIL_TIMEOUT_SECONDS is exceeded.

        参数:
            object_ids: RDT 对象的对象 ID 列表。
            timeout: 用户通过 ray.get 指定的超时时间，None 表示无用户超时。
                实际截止时间是此值与 RDT_FETCH_FAIL_TIMEOUT_SECONDS 的较小值。
            use_object_store: 是否通过对象存储获取，还是通过指定的 Tensor 传输获取。

        返回:
            一个将对象 ID 映射到 RDT 对象（Tensor 列表）的字典。

        异常:
            GetTimeoutError: 如果超过用户指定的超时时间。
            ObjectFetchTimedOutError: 如果超过 RDT_FETCH_FAIL_TIMEOUT_SECONDS。
        """
        from ray.exceptions import GetTimeoutError, ObjectFetchTimedOutError

        rdt_timeout = ray_constants.RDT_FETCH_FAIL_TIMEOUT_SECONDS
        now = time.time()
        if timeout is not None and timeout >= 0:
            rdt_deadline = now + rdt_timeout
            user_deadline = now + timeout
            if user_deadline < rdt_deadline:
                deadline = user_deadline
                user_timeout_is_smaller = True
            else:
                deadline = rdt_deadline
                user_timeout_is_smaller = False
        else:
            deadline = now + rdt_timeout
            user_timeout_is_smaller = False

        rdt_store = self.rdt_store
        result: Dict[str, List[Any]] = {}

        # First, try to get objects that are already available in the store
        # These are primary copies, or secondary copies created via
        # __ray_recv__ that haven't been consumed yet.
        # 首先，尝试获取存储中已经可用的对象
        # 这些是主副本，或通过 __ray_recv__ 创建且尚未被消费的次副本。
        if not use_object_store:
            for object_id in object_ids:
                try:
                    result[object_id] = rdt_store.wait_and_get_object(
                        object_id, timeout=0
                    )
                except TimeoutError:
                    pass

        # For remaining objects, trigger fetches.
        # 对于剩余的对象，触发获取操作。
        fetch_requests: Dict[str, "FetchRequest"] = {}
        for object_id in object_ids:
            if object_id in result:
                continue
            assert self.is_managed_object(
                object_id
            ), f"No metadata found for {object_id}"

            fetch_requests[object_id] = self._trigger_fetch(object_id, use_object_store)

        # Wait for all in-flight fetches to complete.
        # 等待所有进行中的获取操作完成。
        while fetch_requests:
            object_id, fetch_request = fetch_requests.popitem()
            remaining = deadline - time.time()
            if remaining < 0:
                if user_timeout_is_smaller:
                    # User passed a timeout to ray.get that expired.
                    # 用户传入了 ray.get 的超时时间，且已过期。
                    raise GetTimeoutError(f"ray.get timed out after {timeout}s.")
                else:
                    # Object fetch timeout expired. Throw an error in case we
                    # hung.
                    # 对象获取超时已过期。抛出错误以防挂起。
                    raise ObjectFetchTimedOutError(
                        object_ref_hex=object_id,
                        owner_address="",
                        call_site="",
                    )
            try:
                result[object_id] = self._wait_fetch(
                    object_id, fetch_request, timeout=remaining
                )
            except (TimeoutError, GetTimeoutError):
                if user_timeout_is_smaller:
                    raise GetTimeoutError(f"ray.get timed out after {timeout}s.")
                else:
                    raise ObjectFetchTimedOutError(
                        object_ref_hex=object_id,
                        owner_address="",
                        call_site="",
                    )
        return result

    def queue_or_free_object_primary_copy(self, object_id: str):
        """
        Free the RDT object on the primary copy holder and free metadata
        if the tensor metadata is available (the object has been created).
        Otherwise, queue up the free operation until the tensor metadata is available.

        在主副本持有者上释放 RDT 对象，如果 Tensor 元数据可用（对象已创建）则释放元数据。
        否则，将释放操作排队等待 Tensor 元数据可用。
        """
        # NOTE: This may have to change if we support lineage reconstruction for RDT
        # TODO(#57962): Metadata is currently not removed on borrowers that borrow through
        # the NIXL ray.put / ray.get
        # 注意：如果我们为 RDT 支持血统重建，这可能需要变更
        # TODO(#57962): 当前通过 NIXL ray.put / ray.get 借用的借用方不会移除元数据
        with self._lock:
            self._queued_transfers.pop(object_id, None)
            rdt_meta = self._managed_rdt_metadata[object_id]
            tensor_transport_meta = rdt_meta.tensor_transport_meta
            if tensor_transport_meta is None:
                # The object hasn't been created at the time of the free.
                # 在释放时，对象尚未创建。
                self._queued_frees.add(object_id)

        if tensor_transport_meta is not None:
            self.free_object_primary_copy(object_id)

    def free_object_primary_copy(self, object_id: str):
        from ray.experimental.rdt.rdt_store import (
            __ray_free__,
        )

        with self._lock:
            rdt_meta = self._managed_rdt_metadata.pop(object_id)
        src_actor = rdt_meta.src_actor
        tensor_transport_backend = rdt_meta.tensor_transport_backend
        tensor_transport_meta = rdt_meta.tensor_transport_meta
        src_actor.__ray_call__.options(concurrency_group="_ray_system").remote(
            __ray_free__,
            object_id,
            tensor_transport_backend,
            tensor_transport_meta,
        )

    @staticmethod
    def actor_has_tensor_transport(
        actor: "ray.actor.ActorHandle", tensor_transport: str
    ):
        """
        Check if the actor has a communicator for the given tensor transport backend.

        检查 Actor 是否具有给定 Tensor 传输后端的通信器。

        Args:
            actor: The actor to check.
            tensor_transport: The tensor transport backend to check.

        Returns:
            True if the actor has a communicator for the given tensor transport backend, False otherwise.

        参数:
            actor: 要检查的 Actor。
            tensor_transport: 要检查的 Tensor 传输后端。

        返回:
            如果 Actor 具有给定 Tensor 传输后端的通信器则返回 True，否则返回 False。
        """
        from ray.experimental.rdt.util import (
            get_tensor_transport_manager,
        )

        tensor_transport_manager = get_tensor_transport_manager(tensor_transport)
        return tensor_transport_manager.actor_has_tensor_transport(actor)

    def put_object(
        self,
        obj_ref: ObjectRef,
        tensor_transport: str,
        tensors: List[Any],
    ):
        """
        Put the RDT object into the RDT manager.

        将 RDT 对象放入 RDT 管理器。

        Args:
            obj_ref: The object ref of the RDT object.
            tensor_transport: The tensor transport backend to use.
            tensors: The tensors to put into the RDT manager.

        参数:
            obj_ref: RDT 对象的对象引用。
            tensor_transport: 要使用的 Tensor 传输后端。
            tensors: 要放入 RDT 管理器的 Tensor。
        """
        src_actor = ray.get_runtime_context().current_actor
        tensor_transport_meta = self.rdt_store.add_object_primary(
            obj_ref.hex(), tensors, tensor_transport
        )
        self.add_rdt_ref(
            obj_ref,
            src_actor,
            tensor_transport,
            tensor_transport_meta=tensor_transport_meta,
        )
