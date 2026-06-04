import threading
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import ray
from ray._raylet import ObjectRef
from ray.experimental.rdt.collective_tensor_transport import (
    GLOOTensorTransport,
    NCCLTensorTransport,
)
from ray.experimental.rdt.cuda_ipc_transport import CudaIpcTransport
from ray.experimental.rdt.nixl_tensor_transport import (
    NixlTensorTransport,
)
from ray.experimental.rdt.tensor_transport_manager import (
    TensorTransportManager,
    TensorTransportMetadata,
)
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import torch


class TransportManagerInfo(NamedTuple):
    # Class that implements TensorTransportManager
    # 实现 TensorTransportManager 的类
    transport_manager_class: type[TensorTransportManager]
    # List of supported device types for the transport
    # 该传输支持的设备类型列表
    devices: List[str]
    # Data type for this transport (e.g. torch.Tensor or jax.Array)
    # 该传输的数据类型（例如 torch.Tensor 或 jax.Array）
    # If not provided, defaults to torch.Tensor
    # 如果未提供，默认为 torch.Tensor
    data_type: type


transport_manager_info: Dict[str, TransportManagerInfo] = {}

# Singleton instances of transport managers
# 传输管理器的单例实例
transport_managers: Dict[str, TensorTransportManager] = {}

# To protect the singleton instances of transport managers
# 用于保护传输管理器的单例实例
transport_managers_lock = threading.Lock()

# Flipped to True when the first custom transport is registered.
# 当第一个自定义传输注册时翻转为 True。
has_custom_transports = False


@PublicAPI(stability="alpha")
def register_tensor_transport(
    transport_name: str,
    devices: List[str],
    transport_manager_class: type[TensorTransportManager],
    data_type: type,
):
    """
    Register a new tensor transport for use in Ray. Note that this needs to be called
    before you create the actors that will use the transport. The actors also
    need to be created in the same process from which you call this function.

    在 Ray 中注册一个新的张量传输。注意，此函数需要在创建使用该传输的 actor 之前调用。
    这些 actor 还需要在调用此函数的同一进程中创建。

    Args:
        transport_name: The name of the transport protocol.
            传输协议的名称。
        devices: List of PyTorch device types supported by this transport (e.g., ["cuda", "cpu"]).
            该传输支持的 PyTorch 设备类型列表（例如 ["cuda", "cpu"]）。
        transport_manager_class: A class that implements TensorTransportManager.
            实现 TensorTransportManager 的类。
        data_type: The data type for this transport (e.g. torch.Tensor or jax.Array).
            该传输的数据类型（例如 torch.Tensor 或 jax.Array）。
    Raises:
        ValueError: If transport_manager_class is not a subclass of TensorTransportManager.
            如果 transport_manager_class 不是 TensorTransportManager 的子类。
    """
    global transport_manager_info
    global has_custom_transports

    transport_name = transport_name.upper()

    if transport_name in transport_manager_info:
        raise ValueError(f"Transport {transport_name} already registered.")

    if not issubclass(transport_manager_class, TensorTransportManager):
        raise ValueError(
            f"transport_manager_class {transport_manager_class.__name__} must be a subclass of TensorTransportManager."
        )

    transport_manager_info[transport_name] = TransportManagerInfo(
        transport_manager_class, devices, data_type
    )

    if transport_name not in DEFAULT_TRANSPORTS:
        has_custom_transports = True


DEFAULT_TRANSPORTS = ["NIXL", "GLOO", "NCCL", "CUDA_IPC"]

_default_transports_registered = False


def _ensure_default_transports_registered():
    global _default_transports_registered
    with transport_managers_lock:
        if _default_transports_registered:
            return
        _default_transports_registered = True
        try:
            import torch

            register_tensor_transport(
                "NIXL", ["cuda", "cpu"], NixlTensorTransport, torch.Tensor
            )
            register_tensor_transport(
                "GLOO", ["cpu"], GLOOTensorTransport, torch.Tensor
            )
            register_tensor_transport(
                "NCCL", ["cuda"], NCCLTensorTransport, torch.Tensor
            )
            register_tensor_transport(
                "CUDA_IPC", ["cuda"], CudaIpcTransport, torch.Tensor
            )
        except ImportError:
            pass


def get_transport_data_type(tensor_transport: str) -> type:
    _ensure_default_transports_registered()
    if tensor_transport not in transport_manager_info:
        raise ValueError(f"Unsupported tensor transport protocol: {tensor_transport}")

    return transport_manager_info[tensor_transport].data_type


def get_tensor_transport_manager(
    transport_name: str,
) -> "TensorTransportManager":
    """Get the tensor transport manager for the given tensor transport protocol.

    获取给定张量传输协议的张量传输管理器。

    Args:
        transport_name: The tensor transport protocol to use for the GPU object.
            用于 GPU 对象的张量传输协议。

    Returns:
        TensorTransportManager: The tensor transport manager for the given tensor transport protocol.
            给定张量传输协议的张量传输管理器。
    """
    global transport_manager_info
    global transport_managers
    global transport_managers_lock

    _ensure_default_transports_registered()
    with transport_managers_lock:
        if transport_name in transport_managers:
            return transport_managers[transport_name]

        if transport_name not in transport_manager_info:
            raise ValueError(f"Unsupported tensor transport protocol: {transport_name}")

        transport_managers[transport_name] = transport_manager_info[
            transport_name
        ].transport_manager_class()
        return transport_managers[transport_name]


def register_custom_tensor_transports_on_actor(
    actor: "ray.actor.ActorHandle",
) -> Optional[ObjectRef]:
    """
    If there's no custom transports to register, returns None.
    Otherwise returns an ObjectRef for a task on the actor that will register the custom transports.

    如果没有自定义传输需要注册，返回 None。
    否则返回一个 ObjectRef，指向 actor 上将注册自定义传输的任务。
    """
    global transport_manager_info
    global has_custom_transports

    _ensure_default_transports_registered()
    if not has_custom_transports:
        return None

    def register_transport_on_actor(
        self, owner_transport_manager_info: Dict[str, TransportManagerInfo]
    ):
        from ray.experimental.rdt.util import (
            _ensure_default_transports_registered,
            register_tensor_transport,
            transport_manager_info,
        )

        _ensure_default_transports_registered()
        for transport_name, transport_info in owner_transport_manager_info.items():
            if transport_name not in transport_manager_info:
                register_tensor_transport(
                    transport_name,
                    transport_info.devices,
                    transport_info.transport_manager_class,
                    transport_info.data_type,
                )

    return actor.__ray_call__.options(concurrency_group="_ray_system").remote(
        register_transport_on_actor, transport_manager_info
    )


def device_match_transport(device: str, tensor_transport: str) -> bool:
    """Check if the device matches the transport.

    检查设备是否与传输匹配。
    """
    _ensure_default_transports_registered()
    if tensor_transport not in transport_manager_info:
        raise ValueError(f"Unsupported tensor transport protocol: {tensor_transport}")

    return device in transport_manager_info[tensor_transport].devices


def normalize_and_validate_tensor_transport(tensor_transport: str) -> str:
    _ensure_default_transports_registered()
    tensor_transport = tensor_transport.upper()

    if tensor_transport not in transport_manager_info:
        raise ValueError(f"Invalid tensor transport: {tensor_transport}")

    return tensor_transport


def is_one_sided_transport(tensor_transport: str) -> bool:
    _ensure_default_transports_registered()
    return transport_manager_info[
        tensor_transport
    ].transport_manager_class.is_one_sided()


@PublicAPI(stability="alpha")
def register_nixl_memory(tensor: "torch.Tensor") -> None:
    """Registers the tensor's memory with NIXL and bumps the reference count so the memory region is never deregistered.

    将张量的内存注册到 NIXL 并增加引用计数，使该内存区域永远不会被取消注册。

    By default, the lifetime of the NIXL memory registration is tied to the ObjectRef. This means that only when the ObjectRef is created
    do we register the memory with NIXL and deregister it when the ObjectRef goes out of scope. However, this function can be used
    to pre-register a tensor's memory with NIXL and keep it registered for the lifetime of the process which can improve performance
    if the same tensor is re-used in multiple RDT objects.

    默认情况下，NIXL 内存注册的生命周期与 ObjectRef 绑定。这意味着只有在 ObjectRef 创建时才将内存注册到 NIXL，
    并在 ObjectRef 超出作用域时取消注册。然而，此函数可用于预注册张量的内存到 NIXL 并在整个进程生命周期内保持注册，
    这可以在同一张量被多个 RDT 对象重复使用时提升性能。

    If called on a tensor that is already registered with NIXL, we still prevent the tensor's memory from being deregistered.

    如果对已注册到 NIXL 的张量调用此函数，我们仍会阻止该张量的内存被取消注册。

    Args:
        tensor: A PyTorch tensor whose memory should be registered with NIXL.
            应注册到 NIXL 的 PyTorch 张量。

    Example:

        .. code-block:: python

            import torch
            import ray
            from ray.experimental import register_nixl_memory

            @ray.remote(num_gpus=1, enable_tensor_transport=True)
            class Trainer:
                def __init__(self):
                    self.weight = torch.randn(1000, 1000, device="cuda")
                    # Pre-register the memory with NIXL for the lifetime of the process
                    # 在进程生命周期内预注册内存到 NIXL
                    register_nixl_memory(self.weight)

                # Both of the below methods will use the cached NIXL memory registration on multiple calls. You can also mix them,
                # i.e. call get_weight_ref_by_rows then get_weight_ref and get_weight_ref will not trigger a new NIXL memory registration.
                # 以下两个方法在多次调用时将使用缓存的 NIXL 内存注册。你也可以混合使用它们，
                # 即先调用 get_weight_ref_by_rows 再调用 get_weight_ref，get_weight_ref 不会触发新的 NIXL 内存注册。

                # You can ray.put views to each row of the weight matrix if you want to use them separately in your code
                # 你可以 ray.put 权重矩阵每一行的视图，以便在代码中单独使用它们
                def get_weight_ref_by_rows(self):
                    views = [self.weight[i] for i in range(1000)]
                    # Each put call does not trigger a new NIXL memory registration
                    # 每次 put 调用不会触发新的 NIXL 内存注册
                    return ray.put(views, _tensor_transport="nixl")

                # You can also ray.put the entire weight matrix at once
                # 你也可以一次性 ray.put 整个权重矩阵
                def get_weight_ref(self):
                    return ray.put(self.weight, _tensor_transport="nixl")
    """
    nixl_transport = get_tensor_transport_manager("NIXL")
    nixl_transport.register_nixl_memory(tensor)


@PublicAPI(stability="alpha")
def deregister_nixl_memory(tensor: "torch.Tensor") -> None:
    """Decrements the reference count for the tensor's NIXL memory registration added by :func:`ray.experimental.register_nixl_memory`.

    减少由 :func:`ray.experimental.register_nixl_memory` 添加的张量 NIXL 内存注册的引用计数。

    If the reference count reaches 0, the memory is deregistered from NIXL.
    This should only be called after :func:`ray.experimental.register_nixl_memory` has been called for this tensor.
    Any existing ``ray.ObjectRef`` instances that reference this tensor's memory will keep the
    NIXL memory registration alive independently until they go out of scope.

    如果引用计数降至 0，内存将从 NIXL 取消注册。
    此函数只应在 :func:`ray.experimental.register_nixl_memory` 对该张量调用之后调用。
    任何引用此张量内存的现有 ``ray.ObjectRef`` 实例将独立保持 NIXL 内存注册存活，直到它们超出作用域。

    Args:
        tensor: A PyTorch tensor whose NIXL memory registration reference count should be decremented.
            应减少 NIXL 内存注册引用计数的 PyTorch 张量。

    Example:

        .. code-block:: python

            # Extending the example from register_nixl_memory:
            # 扩展 register_nixl_memory 的示例：
            @ray.remote(num_gpus=1, enable_tensor_transport=True)
            class Trainer:
                def deregister_weight(self):
                    # Remove the NIXL memory registration added by register_nixl_memory.
                    # 移除由 register_nixl_memory 添加的 NIXL 内存注册。
                    # The memory may still be registered if there are live ObjectRefs
                    # that reference it.
                    # 如果仍有引用该内存的活跃 ObjectRef，内存可能仍处于注册状态。
                    deregister_nixl_memory(self.weight)
    """
    nixl_transport = get_tensor_transport_manager("NIXL")
    nixl_transport.deregister_nixl_memory(tensor)


@PublicAPI(stability="alpha")
def register_nixl_memory_pool(size: int, device: "torch.device") -> None:
    """Pre-allocates a memory pool and registers it with NIXL.

    预分配内存池并将其注册到 NIXL。

    This enables pool-based memory management for NIXL transfers, which can improve
    performance by avoiding repeated memory registration/deregistration. The pool is
    registered once with NIXL and individual tensors are copied into it on ``ray.put``.

    这为 NIXL 传输启用了基于池的内存管理，可通过避免重复的内存注册/取消注册来提升性能。
    该池一次性注册到 NIXL，各个张量在 ``ray.put`` 时被复制到池中。

    Within a single ``ray.put`` call, tensors sharing the same underlying storage
    (including views) are automatically deduplicated — only one copy of each unique
    storage is allocated. Across multiple ``ray.put`` calls, if the same storage
    appears again, the existing pool slot is reused without re-copying the data.
    As a result, data can be potentially stale once you ``ray.put`` the storage
    tensor — subsequent mutations to that storage may not be reflected in outstanding refs.
    Clone the tensor before ``ray.put`` if snapshot semantics are required.

    在单次 ``ray.put`` 调用中，共享相同底层存储（包括视图）的张量会自动去重——
    每个唯一存储只分配一份副本。在多次 ``ray.put`` 调用中，如果相同存储再次出现，
    现有的池槽位会被复用而无需重新复制数据。因此，一旦你 ``ray.put`` 了存储张量，
    数据可能已过时——对该存储的后续修改可能不会反映在已有的引用中。
    如果需要快照语义，请在 ``ray.put`` 前克隆张量。

    If the pool has insufficient space for an allocation,
    :class:`NixlOutOfMemoryError` is raised.

    如果池中没有足够的空间用于分配，将抛出 :class:`NixlOutOfMemoryError`。

    Args:
        size: Size of the memory pool in bytes.
            内存池的大小，以字节为单位。
        device: Device to allocate the pool on (e.g., ``torch.device("cpu")``
            or ``torch.device("cuda")``).
            分配内存池的设备（例如 ``torch.device("cpu")`` 或 ``torch.device("cuda")``）。

    Example:

        .. code-block:: python

            import torch
            import ray
            from ray.experimental import register_nixl_memory_pool

            @ray.remote(num_gpus=1, enable_tensor_transport=True)
            class Trainer:
                def __init__(self):
                    # Pre-allocate a 1GB GPU memory pool for NIXL transfers
                    # 预分配一个 1GB 的 GPU 内存池用于 NIXL 传输
                    register_nixl_memory_pool(1024 * 1024 * 1024, torch.device("cuda"))

                def get_weight_ref(self):
                    weight = torch.randn(1000, 1000, device="cuda")
                    return ray.put(weight, _tensor_transport="nixl")
    """
    nixl_transport = get_tensor_transport_manager("NIXL")
    nixl_transport.register_nixl_memory_pool(size, device)


def create_empty_tensors_from_metadata(
    tensor_transport_meta: TensorTransportMetadata,
) -> List["torch.Tensor"]:
    import torch

    tensors = []
    device = tensor_transport_meta.tensor_device
    for meta in tensor_transport_meta.tensor_meta:
        shape, dtype = meta
        tensor = torch.empty(shape, dtype=dtype, device=device)
        tensors.append(tensor)
    return tensors
