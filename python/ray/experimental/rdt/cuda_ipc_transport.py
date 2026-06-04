from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import ray
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)

if TYPE_CHECKING:
    import torch


@dataclass
class CudaIpcCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the CUDA IPC communicator.
    CUDA IPC 通信器的元数据。"""


@dataclass
class CudaIpcTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for CUDA IPC transport.
    存储在 GPU 对象存储中用于 CUDA IPC 传输的 Tensor 的元数据。"""

    # List of tuples, each containing the function and metadata to reconstruct the tensor.
    # 元组列表，每个元组包含用于重建 Tensor 的函数和元数据。
    cuda_ipc_handles: Optional[List[Any]] = None
    # The IPC handle of the event that is used to synchronize the sender and receiver.
    # 用于同步发送方和接收方的事件的 IPC handle。
    cuda_ipc_event_ipc_handle: Optional[bytes] = None
    # The index of the GPU that the tensors are on. This requires that the GPU is
    # assigned by Ray, e.g., using @ray.remote(num_gpus=1).
    # Tensor 所在 GPU 的索引。这要求 GPU 由 Ray 分配，例如使用 @ray.remote(num_gpus=1)。
    ray_gpu_idx: Optional[int] = None
    # The node that the GPU that the tensors are on is on.
    # Tensor 所在 GPU 所属的节点。
    ray_node_id: Optional[str] = None


class CudaIpcTransport(TensorTransportManager):
    def __init__(self):
        pass

    @property
    def tensor_transport_backend(self) -> str:
        return "CUDA_IPC"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        # TODO: Ideally we would check if torch.cuda.is_available() on the actor
        # and if so, return True. But we want to avoid blocking in ray.get() in
        # this method since it gets called before submitting an actor task.
        # TODO: 理想情况下，我们会检查 actor 上 torch.cuda.is_available() 是否可用，
        # 如果可用则返回 True。但我们希望避免在此方法中使用 ray.get() 造成阻塞，
        # 因为该方法在提交 actor 任务之前被调用。
        return True

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        rdt_object: List["torch.Tensor"],
    ) -> CudaIpcTransportMetadata:

        tensor_meta = []
        device = None
        cuda_ipc_handles = []
        event_ipc_handle = None
        ray_gpu_idx = None
        ray_node_id = None
        if rdt_object:
            import torch
            from torch.multiprocessing.reductions import reduce_tensor

            device = rdt_object[0].device
            ray_gpu_idx = ray.get_gpu_ids()[device.index]
            ray_node_id = ray.get_runtime_context().get_node_id()

            # Create an interprocess-shareable CUDA event so that the receiver
            # can wait for the sender's computations to complete.
            # 创建一个可跨进程共享的 CUDA event，以便接收方可以等待发送方的计算完成。
            event = torch.cuda.Event(interprocess=True)
            torch.cuda.current_stream(device).record_event(event)

            for t in rdt_object:
                if t.device.type != device.type:
                    raise ValueError(
                        "All tensors in an RDT object must have the same device type."
                    )
                if t.device.index != device.index:
                    raise ValueError(
                        "All tensors in an RDT object must be on the same GPU."
                    )
                tensor_meta.append((t.shape, t.dtype))
                ipc_handle = reduce_tensor(t)
                cuda_ipc_handles.append(ipc_handle)

            event_ipc_handle = event.ipc_handle()

        return CudaIpcTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=device.type if device else None,
            cuda_ipc_handles=cuda_ipc_handles,
            cuda_ipc_event_ipc_handle=event_ipc_handle,
            ray_gpu_idx=ray_gpu_idx,
            ray_node_id=ray_node_id,
        )

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> CudaIpcCommunicatorMetadata:

        communicator_metadata = CudaIpcCommunicatorMetadata()
        return communicator_metadata

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> List["torch.Tensor"]:

        assert isinstance(
            tensor_transport_metadata, CudaIpcTransportMetadata
        ), "metadata must be a CudaIpcTransportMetadata object for CUDA IPC transport"
        assert isinstance(
            communicator_metadata, CudaIpcCommunicatorMetadata
        ), "metadata must be a CudaIpcCommunicatorMetadata object for CUDA IPC transport"

        if target_buffers:
            raise ValueError(
                "The CUDA IPC transport does not support receiving into buffers."
            )

        tensors = []
        if tensor_transport_metadata.tensor_meta:
            import torch

            cur_node_id = ray.get_runtime_context().get_node_id()
            if cur_node_id != tensor_transport_metadata.ray_node_id:
                raise ValueError(
                    f"CUDA IPC transport only supports tensors on the same node, but the current node ID: {cur_node_id} and the sender node ID: {tensor_transport_metadata.ray_node_id} are different."
                )

            try:
                device_idx = ray.get_gpu_ids().index(
                    tensor_transport_metadata.ray_gpu_idx
                )
            except ValueError:
                raise ValueError(
                    f"CUDA IPC transport only supports tensors on the same GPU, but the receiver was not allocated the same GPUs by Ray as the sender (GPU: {tensor_transport_metadata.ray_gpu_idx}). To use the CUDA IPC RDT transport, ensure that the receiver is allocated the same GPU by Ray as the sender, and that CUDA_VISIBLE_DEVICES is set to `ray.get_gpu_ids()`, the GPUs assigned by Ray (this is the default behavior)."
                )
            device = torch.device(f"cuda:{device_idx}")

            event_ipc_handle = tensor_transport_metadata.cuda_ipc_event_ipc_handle
            if event_ipc_handle is not None:
                # Reconstruct the event from IPC handle
                # 从 IPC handle 重建事件
                event_remote = torch.cuda.Event.from_ipc_handle(
                    device=device, handle=event_ipc_handle
                )

                # Make current stream wait for the sender's event
                # This ensures sender's computation is complete before we use the tensor
                # This is asynchronous - doesn't block CPU, only GPU stream
                # 使当前 stream 等待发送方的 event
                # 这确保在我们使用 Tensor 之前发送方的计算已经完成
                # 这是异步操作 - 不阻塞 CPU，仅阻塞 GPU stream
                torch.cuda.current_stream(device).wait_event(event_remote)

            for i, ipc_handle in enumerate(tensor_transport_metadata.cuda_ipc_handles):
                # Reconstruct the tensor
                # 重建 Tensor
                func, args = ipc_handle
                list_args = list(args)
                # Fields specified in https://github.com/pytorch/pytorch/blob/1495b35d29512f303ab37780760c5e692158514b/torch/multiprocessing/reductions.py#L155
                # 字段定义参见上述 PyTorch 源码链接
                # Update device ID to match current process's device mapping
                # 更新 device ID 以匹配当前进程的 device 映射
                if not isinstance(list_args[6], int):
                    raise RuntimeError(
                        f"Expected CUDA IPC tensor reconstruction list_args[6] to be device ID, but got {list_args[6]}. Please file an issue at https://github.com/ray-project/ray/issues/new/choose."
                    )
                list_args[6] = device.index
                try:
                    tensor = func(*list_args)
                except Exception as e:
                    raise RuntimeError(
                        "Error reconstructing CUDA IPC tensor. Source actor may have failed."
                    ) from e
                tensors.append(tensor)
        return tensors

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: CudaIpcTransportMetadata,
        communicator_metadata: CudaIpcCommunicatorMetadata,
    ):
        raise NotImplementedError(
            "CUDA IPC transport does not support send_multiple_tensors, since it is a one-sided transport."
        )

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: CudaIpcTransportMetadata,
        tensors: List["torch.Tensor"],
    ):
        pass

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CudaIpcCommunicatorMetadata,
    ):
        # TODO: Implement CUDA IPC abort transport.
        # TODO: 实现 CUDA IPC 中止传输功能。
        raise NotImplementedError(
            "CUDA IPC transport does not support abort_transport for now."
        )
