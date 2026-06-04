from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    import torch

    import ray


# NOTE: This is a public facing abstract interface for custom tensor transports.
# 注意：这是用于自定义张量传输的公开抽象接口。
# Be sure to update the direct-transport docs when making changes to this interface, especially if changing the path to the file.
# 修改此接口时务必更新 direct-transport 文档，尤其是更改文件路径时。


@dataclass
class CommunicatorMetadata:
    """Metadata for the communicator.

    通信器的元数据。"""


@dataclass
class TensorTransportMetadata:
    """Metadata for tensors stored in the GPU object store.

    存储在 GPU 对象存储中的张量元数据。

    Args:
        tensor_meta: A list of tuples, each containing the shape and dtype of a tensor.
            tensor_meta: 一个元组列表，每个元组包含一个张量的形状和 dtype。
        tensor_device: The device of the tensor. Currently, we require all tensors in the
        list have the same device type.
            tensor_device: 张量的设备。目前要求列表中所有张量具有相同的设备类型。
    """

    tensor_meta: List[
        Union[Tuple["torch.Size", "torch.dtype"], Tuple[Tuple[int, ...], "np.dtype"]]
    ]
    tensor_device: Optional[str] = None


@dataclass
class FetchRequest:
    """Represents a pending or completed tensor fetch operation.

    表示一个正在进行或已完成的张量拉取操作。

    The default fetch/wait implementation stores the tensors here directly
    after a synchronous recv. Transports with true async capability may
    subclass this to carry additional state needed by wait_fetch_complete.

    默认的 fetch/wait 实现在同步 recv 后直接将张量存储在此处。
    具有真正异步能力的传输可以子类化此项，以携带 wait_fetch_complete 所需的额外状态。

    Subclasses should handle all resource cleanup in __del__ rather than
    in wait_fetch_complete, so that resources are released even if the
    caller never waits on the request.

    子类应在 __del__ 中而非 wait_fetch_complete 中处理所有资源清理，
    以确保即使调用者从未等待该请求，资源也能被释放。

    Args:
        obj_id: The object ID for the fetch operation.
            obj_id: 拉取操作的对象 ID。
        tensors: The fetched tensors.
            tensors: 已拉取的张量。
    """

    obj_id: str
    tensors: List[Any]


class TensorTransportManager(ABC):
    """
    Interface with which to implement custom tensor transports.

    用于实现自定义张量传输的接口。
    """

    @abstractmethod
    def tensor_transport_backend(self) -> str:
        """
        Returns the name of your tensor transport backend.
        Ray uses this name to match your transport with the ``tensor_transport`` argument
        on the method.

        返回你的张量传输后端的名称。
        Ray 使用此名称将你的传输与方法上的 ``tensor_transport`` 参数进行匹配。

        Returns:
            str: The backend of the tensor transport.
                str: 张量传输的后端名称。
        """

    @staticmethod
    @abstractmethod
    def is_one_sided() -> bool:
        """
        Indicates whether your transport uses one-sided communication where only the receiver
        initiates the transfer.

        指示你的传输是否使用单侧通信，即仅由接收方发起传输。

        One-sided transports: The receiver can directly read the sender's memory without the sender
        actively participating. NIXL and CUDA-IPC are examples.

        单侧传输：接收方可以直接读取发送方的内存，无需发送方主动参与。
        NIXL 和 CUDA-IPC 是此类示例。

        Two-sided transports: Both sender and receiver must actively participate in the transfer.
        Collective communication libraries like NCCL and GLOO are examples.

        双侧传输：发送方和接收方都必须主动参与传输。
        集合通信库如 NCCL 和 GLOO 是此类示例。

        This affects how Ray orchestrates the transfer and handles failures. Two-sided transports
        have extra limitations described in :ref:`limitations <limitations>`. Ray will not call
        `send_multiple_tensors` for one-sided transports; the transfer is expected to happen through
        just `recv_multiple_tensors`.

        这影响 Ray 如何编排传输和处理故障。
        双侧传输有额外的限制，见 :ref:`limitations <limitations>`。
        Ray 不会为单侧传输调用 `send_multiple_tensors`；传输预期仅通过 `recv_multiple_tensors` 完成。

        Returns:
            bool: True if the backend is one-sided, False otherwise.
                bool: 如果后端是单侧的返回 True，否则返回 False。
        """

    @staticmethod
    @abstractmethod
    def can_abort_transport() -> bool:
        """
        Indicates whether your transport can safely abort an in-progress transfer.

        指示你的传输是否可以安全中止正在进行中的传输。

        If ``True``, Ray calls `abort_transport` on both the source and destination actors when a
        send / recv error, allowing your transport to clean up gracefully.

        如果为 ``True``，Ray 在发生发送/接收错误时会在源和目标 actor 上调用 `abort_transport`，
        允许你的传输优雅地清理资源。

        If ``False``, Ray kills the involved actors to prevent deadlocks when errors occur during
        transfer.

        如果为 ``False``，Ray 在传输期间发生错误时会终止涉及的 actor，以防止死锁。

        Return ``True`` only if your transport can reliably interrupt an in-progress send or receive
        operation without leaving either party in a blocked state.

        仅当你的传输能够可靠地中断正在进行的发送或接收操作，
        且不会使任一方处于阻塞状态时，才返回 ``True``。

        Returns:
            bool: True if the backend can abort the transport.
                bool: 如果后端可以中止传输则返回 True。
        """

    @abstractmethod
    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Whether the actor has the tensor transport available.

        判断 actor 是否具有可用的张量传输。

        Args:
            actor: The actor to check.
                actor: 要检查的 actor。

        Returns:
            bool: True if the actor has the tensor transport available, False otherwise.
                bool: 如果 actor 具有可用的张量传输返回 True，否则返回 False。
        """

    @abstractmethod
    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        rdt_object: List[Any],
    ) -> TensorTransportMetadata:
        """
        Implement this method to create the TensorTransportMetadata you defined previously.
        Ray calls this on the source actor immediately after the actor task creates the result tensors.
        Implement this to:

        实现此方法以创建你先前定义的 TensorTransportMetadata。
        Ray 在 actor 任务创建结果张量后立即在源 actor 上调用此方法。
        实现此方法以：

        1. Record tensor shapes, dtypes, and devices.
           1. 记录张量的形状、dtype 和设备。
        2. Perform any transport-specific tensor registration such as registering memory for RDMA.
           2. 执行任何传输特定的张量注册，例如为 RDMA 注册内存。
        3. Store any handles or identifiers needed for the transfer.
           3. 存储传输所需的任何句柄或标识符。

        Args:
            obj_id: The ID of the RDT object to extract the tensor transport metadata from.
                obj_id: 要提取张量传输元数据的 RDT 对象的 ID。
            rdt_object: The RDT object to extract the tensor transport metadata from.
                rdt_object: 要提取张量传输元数据的 RDT 对象。

        Returns:
            TensorTransportMetadata: The tensor transport metadata.
                TensorTransportMetadata: 张量传输元数据。
        """

    @abstractmethod
    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> CommunicatorMetadata:
        """
        Gets the CommunicatorMetadata for a send/recv. Ray calls this on the owner/driver process before
        orchestrating the transfer. You can typically implement this to return information both actors
        need to identify each other such as ranks in a collective group. Many forms of transports such
        as one-sided RDMA reads may be ok just returning empty CommunicatorMetadata here.

        获取 send/recv 的 CommunicatorMetadata。
        Ray 在编排传输之前在 owner/driver 进程上调用此方法。
        通常可以实现此方法以返回两个 actor 互相识别所需的信息，
        例如集合通信组中的 rank。
        许多形式的传输（如单侧 RDMA 读取）可以直接返回空的 CommunicatorMetadata。

        Args:
            src_actor: The actor that runs this function.
                src_actor: 运行此函数的源 actor。
            dst_actor: The actor that runs this function.
                dst_actor: 运行此函数的目标 actor。
            backend: The backend to use for the collective operation.
                backend: 用于集合操作的后端。

        Returns:
            CommunicatorMetadata: The communicator metadata.
                CommunicatorMetadata: 通信器元数据。
        """

    @abstractmethod
    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List[Any]] = None,
    ) -> List[Any]:
        """
        Receives tensors on the destination actor. Ray calls this on the destination
        actor during the transfer.

        在目标 actor 上接收张量。Ray 在传输期间在目标 actor 上调用此方法。

        Args:
            obj_id: The object ID for related GPU object.
                obj_id: 相关 GPU 对象的对象 ID。
            tensor_transport_metadata: The tensor transport metadata for the GPU object.
                tensor_transport_metadata: GPU 对象的张量传输元数据。
            communicator_metadata: The communicator metadata for the send/recv operation.
                communicator_metadata: send/recv 操作的通信器元数据。
            target_buffers: Pre-allocated buffers to receive the tensors into if possible.
                target_buffers: 预分配的缓冲区，用于尽可能将张量接收进来。
        Returns:
            List[Any]: The received tensors.
                List[Any]: 已接收的张量。
        """

    def fetch_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List[Any]] = None,
    ) -> FetchRequest:
        """Initiate a fetch for multiple tensors without waiting for completion.

        发起多个张量的拉取，无需等待完成。

        The default implementation calls recv_multiple_tensors synchronously and
        stores the result in a FetchRequest. Transports with true async capability
        should override both this method and wait_fetch_complete.

        默认实现同步调用 recv_multiple_tensors并将结果存储在 FetchRequest 中。
        具有真正异步能力的传输应同时覆盖此方法和 wait_fetch_complete。

        Call wait_fetch_complete(fetch_request) afterward to retrieve the tensors.

        之后调用 wait_fetch_complete(fetch_request) 来获取张量。

        Args:
            obj_id: The object ID for the related GPU object.
                obj_id: 相关 GPU 对象的对象 ID。
            tensor_transport_metadata: The tensor transport metadata for the GPU object.
                tensor_transport_metadata: GPU 对象的张量传输元数据。
            communicator_metadata: The communicator metadata for the send/recv operation.
                communicator_metadata: send/recv 操作的通信器元数据。
            target_buffers: Pre-allocated buffers to receive the tensors into if possible.
                target_buffers: 预分配的缓冲区，用于尽可能将张量接收进来。

        Returns:
            A FetchRequest whose tensors field is already populated.
                一个 FetchRequest，其 tensors 字段已填充完毕。
        """
        tensors = self.recv_multiple_tensors(
            obj_id, tensor_transport_metadata, communicator_metadata, target_buffers
        )
        return FetchRequest(obj_id=obj_id, tensors=tensors)

    def wait_fetch_complete(
        self, fetch_request: FetchRequest, timeout: float = -1
    ) -> List[Any]:
        """Wait for a previously initiated fetch to complete and return the tensors.

        等待先前发起的拉取完成并返回张量。

        The default implementation returns the tensors stored in the FetchRequest
        directly, since the default fetch_multiple_tensors is synchronous.

        默认实现直接返回 FetchRequest 中存储的张量，
        因为默认的 fetch_multiple_tensors 是同步的。

        Args:
            fetch_request: The FetchRequest returned by fetch_multiple_tensors.
                fetch_request: 由 fetch_multiple_tensors 返回的 FetchRequest。
            timeout: Maximum time in seconds to wait. -1 means wait indefinitely.
                0 means return immediately if not ready.
                timeout: 最大等待时间（秒）。-1 表示无限等待。
                    0 表示如果未就绪则立即返回。

        Returns:
            The received tensors.
                已接收的张量。

        Raises:
            TimeoutError: If timeout is exceeded.
                TimeoutError: 如果超时。
        """
        return fetch_request.tensors

    @abstractmethod
    def send_multiple_tensors(
        self,
        tensors: List[Any],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        """
        Sends tensors from the source actor to the destination actor. Ray calls this on the source actor
        during the transfer. Implement this to perform the actual data transfer using your transport's
        send mechanism. For one-sided transports, you can simply avoid implementing this method or even
        raise a NotImplementedError to ensure it's not being called.

        从源 actor 向目标 actor 发送张量。Ray 在传输期间在源 actor 上调用此方法。
        实现此方法以使用你传输的发送机制执行实际的数据传输。
        对于单侧传输，可以简单地不实现此方法，
        或者甚至抛出 NotImplementedError 以确保它不会被调用。

        Args:
            tensors: The tensors or jax arrays to send.
                tensors: 要发送的张量或 jax 数组。
            tensor_transport_metadata: The tensor transport metadata for the RDT object.
                tensor_transport_metadata: RDT 对象的张量传输元数据。
            communicator_metadata: The communicator metadata for the send/recv operation.
                communicator_metadata: send/recv 操作的通信器元数据。
        """

    @abstractmethod
    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: List[Any],
    ):
        """
        Clean up resources for an RDT object. Ray calls this on the source actor
        after Ray's distributed reference counting protocol determines the object is out of scope.

        清理 RDT 对象的资源。
        Ray 在其分布式引用计数协议确定对象超出作用域后，在源 actor 上调用此方法。

        Use this to release any resources your transport allocated, such as deregistering memory buffers.
        On the receiver side, no cleanup is needed — Ray does not hold onto the tensor after
        returning it to the user, so it is garbage collected normally when the user releases it.

        使用此方法释放你的传输分配的任何资源，例如注销内存缓冲区。
        在接收方不需要清理——Ray 在将张量返回给用户后不会持有它，
        因此当用户释放它时会正常进行垃圾回收。

        Args:
            obj_id: The ID of the GPU object to garbage collect.
                obj_id: 要垃圾回收的 GPU 对象的 ID。
            tensor_transport_meta: The tensor transport metadata.
                tensor_transport_meta: 张量传输元数据。
            tensors: The tensors that are contained in the ObjectRef that is being freed.
                tensors: 包含在被释放的 ObjectRef 中的张量。
        """

    @abstractmethod
    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        """
        Aborts an in-progress transfer. Ray calls this on both the source and destination actors
        when a system error occurs if `can_abort_transport` returns ``True``.

        中止正在进行中的传输。
        当系统错误发生且 `can_abort_transport` 返回 ``True`` 时，
        Ray 在源和目标 actor 上都调用此方法。

        Args:
            obj_id: The object ID for related GPU object.
                obj_id: 相关 GPU 对象的对象 ID。
            communicator_metadata: The communicator metadata for the send/recv operation.
                communicator_metadata: send/recv 操作的通信器元数据。
        """
