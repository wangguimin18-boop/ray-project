# Ray Direct Transport (RDT) 代码级深度解析

> 源码位置: `python/ray/experimental/rdt/`
> 文档位置: `doc/source/ray-core/direct-transport/`

---

## 1. 类图 — 核心类继承关系

```mermaid
classDiagram
    direction TB

    class TensorTransportManager {
        <<abstract>>
        +tensor_transport_backend() str
        +is_one_sided() bool$
        +can_abort_transport() bool$
        +actor_has_tensor_transport(actor) bool
        +extract_tensor_transport_metadata(obj_id, rdt_object) TensorTransportMetadata
        +get_communicator_metadata(src_actor, dst_actor, backend) CommunicatorMetadata
        +recv_multiple_tensors(obj_id, meta, comm_meta, target_buffers) List
        +fetch_multiple_tensors(obj_id, meta, comm_meta, target_buffers) FetchRequest
        +wait_fetch_complete(fetch_request, timeout) List
        +send_multiple_tensors(tensors, meta, comm_meta)
        +garbage_collect(obj_id, meta, tensors)
        +abort_transport(obj_id, comm_meta)
    }

    class CommunicatorMetadata {
        <<dataclass>>
    }

    class TensorTransportMetadata {
        <<dataclass>>
        +tensor_meta List~Tuple~
        +tensor_device Optional~str~
    }

    class FetchRequest {
        <<dataclass>>
        +obj_id str
        +tensors List
    }

    class CollectiveTensorTransport {
        +is_one_sided() bool$ False
        +can_abort_transport() bool$ False
        +extract_tensor_transport_metadata(obj_id, rdt_object) CollectiveTransportMetadata
        +get_communicator_metadata(src, dst, backend) CollectiveCommunicatorMetadata
        +recv_multiple_tensors(...) List
        +send_multiple_tensors(...)
    }

    class NCCLTensorTransport {
        +tensor_transport_backend() "NCCL"
    }

    class GLOOTensorTransport {
        +tensor_transport_backend() "GLOO"
    }

    class NixlTensorTransport {
        +is_one_sided() bool$ True
        +can_abort_transport() bool$ True
        -_nixl_agent Any
        -_tensor_desc_cache Dict
        -_managed_meta_nixl Dict
        -_remote_agents OrderedDict
        -_memory_pool MemoryPoolManager
        +get_nixl_agent() nixl_agent
        +register_nixl_memory(tensor)
        +register_nixl_memory_pool(size, device)
        +fetch_multiple_tensors(...) NixlFetchRequest
        +wait_fetch_complete(fetch_request, timeout) List
        +recv_multiple_tensors(...) List
    }

    class CudaIpcTransport {
        +is_one_sided() bool$ True
        +can_abort_transport() bool$ False
        +recv_multiple_tensors(...) List
    }

    class CollectiveCommunicatorMetadata {
        +communicator_name str
        +src_rank Optional~int~
        +dst_rank Optional~int~
    }

    class CollectiveTransportMetadata {
    }

    class NixlCommunicatorMetadata {
    }

    class NixlTransportMetadata {
        +nixl_serialized_descs Optional~bytes~
        +nixl_agent_meta Optional~bytes~
        +nixl_agent_name Optional~str~
        +nixl_agent_meta_version Optional~int~
    }

    class CudaIpcCommunicatorMetadata {
    }

    class CudaIpcTransportMetadata {
        +cuda_ipc_handles Optional~List~
        +cuda_ipc_event_ipc_handle Optional~bytes~
        +ray_gpu_idx Optional~int~
        +ray_node_id Optional~str~
    }

    class NixlFetchRequest {
        +xfer_handle Any
        +nixl_agent Any
        +remote_name Optional~str~
        +remove_tensor_descs bool
        +transport Any
        +__del__()
    }

    class RDTManager {
        -_lock threading.Lock
        -_managed_rdt_metadata Dict~str, RDTMeta~
        -_queued_transfers Dict~str, List~
        -_queued_frees Set~str~
        -_rdt_store Optional~RDTStore~
        -_unmonitored_transfers Queue
        -_monitor_failures_thread Thread
        -actor_id_to_transports_registered Dict
        +add_rdt_ref(obj_ref, src_actor, tensor_transport, meta)
        +set_rdt_metadata(obj_id, rdt_meta)
        +get_rdt_metadata(obj_id) Optional~RDTMeta~
        +trigger_out_of_band_tensor_transfer(dst_actor, obj_id)
        +queue_or_trigger_out_of_band_tensor_transfer(dst_actor, task_args)
        +fetch_and_get_rdt_objects(object_ids, timeout, use_object_store) Dict
        +_trigger_fetch(obj_id, use_object_store) FetchRequest
        +_wait_fetch(obj_id, fetch_request, timeout) List
        +get_rdt_objects(object_ids) Dict
        +put_object(obj_ref, tensor_transport, tensors)
        +free_object_primary_copy(object_id)
        +queue_or_free_object_primary_copy(object_id)
        +set_target_buffers_for_ref(ref, target_buffers)
        +start_monitor_thread_if_needed()
        +shutdown()
    }

    class RDTStore {
        -_rdt_store Dict~str, deque~_RDTObject~~
        -_tensor_to_object_ids Dict~int, Set~str~~
        -_lock threading.RLock
        -_object_present_cv threading.Condition
        -_object_freed_cv threading.Condition
        +has_object(obj_id) bool
        +get_object(obj_id) Optional~List~
        +add_object(obj_id, rdt_object, is_primary)
        +add_object_primary(obj_id, tensors, tensor_transport) TensorTransportMetadata
        +wait_and_get_object(obj_id, timeout) List
        +wait_and_pop_object(obj_id, timeout) List
        +pop_object(obj_id) List
        +wait_tensor_freed(tensor, timeout)
    }

    class RDTMeta {
        <<NamedTuple>>
        +src_actor ActorHandle
        +tensor_transport_backend str
        +tensor_transport_meta Optional~TensorTransportMetadata~
        +sent_dest_actors Set~str~
        +sent_to_src_actor_and_others_warned bool
        +target_buffers Optional~List~
    }

    class TransferMetadata {
        <<NamedTuple>>
        +src_actor ActorHandle
        +dst_actor ActorHandle
        +send_ref Optional~ObjectRef~
        +recv_ref ObjectRef
        +communicator_meta CommunicatorMetadata
        +backend str
        +obj_id str
        +timeout float
    }

    class MemoryPoolManager {
        -_pool_tensor torch.Tensor
        -_free_blocks List~MemoryBlock~
        -_allocated_blocks Dict~int, MemoryBlock~
        +allocate_for_tensors(tensors) List~Tensor~
        +free_tensors(tensors)
        +has_block(tensor) bool
    }

    TensorTransportManager <|-- CollectiveTensorTransport
    TensorTransportManager <|-- NixlTensorTransport
    TensorTransportManager <|-- CudaIpcTransport
    CollectiveTensorTransport <|-- NCCLTensorTransport
    CollectiveTensorTransport <|-- GLOOTensorTransport

    CommunicatorMetadata <|-- CollectiveCommunicatorMetadata
    CommunicatorMetadata <|-- NixlCommunicatorMetadata
    CommunicatorMetadata <|-- CudaIpcCommunicatorMetadata
    TensorTransportMetadata <|-- CollectiveTransportMetadata
    TensorTransportMetadata <|-- NixlTransportMetadata
    TensorTransportMetadata <|-- CudaIpcTransportMetadata
    FetchRequest <|-- NixlFetchRequest

    RDTManager --> RDTStore : owns lazily
    RDTManager --> RDTMeta : manages metadata
    RDTManager --> TransferMetadata : monitors transfers
    RDTStore --> _RDTObject : stores objects
    NixlTensorTransport --> MemoryPoolManager : owns lazily
    NixlTensorTransport --> NixlTransportMetadata : caches _managed_meta_nixl
    NixlTensorTransport --> TensorDesc : caches _tensor_desc_cache
```

---

## 2. 传输分类对比

| 传输后端 | 类型 | `is_one_sided` | `can_abort_transport` | 需要集体组 | 支持设备 | `ray.get` 直接使用 |
|---------|------|----------------|----------------------|-----------|---------|------------------|
| **GLOO** | 双侧 | False | False | 是 | CPU | 否 (需 `_use_object_store=True`) |
| **NCCL** | 双侧 | False | False | 是 | CUDA | 否 (需 `_use_object_store=True`) |
| **NIXL** | 单侧 | True | True | 否 | CPU/CUDA | 是 |
| **CUDA_IPC** | 单侧 | True | False | 否 | CUDA (同节点同GPU) | 是 |

---

## 3. 用户使用流程图 — 双侧传输 (GLOO/NCCL)

```mermaid
flowchart TB
    subgraph UserCode["用户代码"]
        A[定义 Actor 类<br>@ray.method.tensor_transport='nccl'] --> B[创建 Actor 实例<br>a1, a2 = Actor.remote]
        B --> C[创建集合通信组<br>create_collective_group<br>[a1, a2], backend='nccl']
        C --> D[调用返回 tensor 的任务<br>ref = a1.get_tensor.remote]
        D --> E[将 ref 传给另一个 actor<br>a2.consume.remote]
        E --> F[可选: ray.get<br>_use_object_store=True]
    end

    subgraph RayInternals["Ray 内部"]
        D -.-> D1[RDTManager.add_rdt_ref<br>记录 src_actor + transport 类型]
        D1 -.-> D2[Actor 任务返回 tensor<br>RDTStore.add_object_primary]
        D2 -.-> D3[提取 TensorTransportMetadata<br>extract_tensor_transport_metadata]
        D3 -.-> D4[元数据回传到 owner<br>set_tensor_transport_metadata_and_trigger_queued_operations]
        E -.-> E1[queue_or_trigger_out_of_band_tensor_transfer<br>检测 task_args 中的 ObjectRef]
        E1 -.-> E2[get_communicator_metadata<br>获取 src_rank / dst_rank]
        E2 -.-> E3[提交 __ray_send__ 到 src_actor]
        E2 -.-> E4[提交 __ray_recv__ 到 dst_actor]
        E3 -.-> E5[send_multiple_tensors<br>NCCL/GLOO send]
        E4 -.-> E6[recv_multiple_tensors<br>NCCL/GLOO recv]
        E6 -.-> E7[RDTStore.add_object<br>tensor 存入接收方本地 store]
        E7 -.-> E8[反序列化时合并 CPU 数据 + tensor]
        F -.-> F1[fetch_and_get_rdt_objects<br>use_object_store=True]
        F1 -.-> F2[src_actor.__ray_fetch_rdt_object__<br>从 RDTStore 获取 tensor]
        F2 -.-> F3[通过 Ray 对象存储返回]
    end

    style UserCode fill:#e8f5e9
    style RayInternals fill:#e3f2fd
```

---

## 4. 用户使用流程图 — 单侧传输 (NIXL)

```mermaid
flowchart TB
    subgraph UserCode["用户代码"]
        A[定义 Actor 类<br>@ray.method.tensor_transport='nixl'] --> B[创建 Actor 实例<br>a1, a2 = Actor.remote<br>enable_tensor_transport=True]
        B --> C[可选: register_nixl_memory<br>或 register_nixl_memory_pool<br>预注册内存提升性能]
        C --> D[调用返回 tensor 的任务<br>ref = a1.get_tensor.remote]
        D --> E[将 ref 传给另一个 actor<br>a2.consume.remote]
        E --> F[可选: ray.get<br>直接用 NIXL 拉取]
    end

    subgraph RayInternals["Ray 内部"]
        D -.-> D1[RDTManager.add_rdt_ref<br>记录 src_actor + 'NIXL']
        D1 -.-> D2[Actor 任务返回 tensor<br>RDTStore.add_object_primary]
        D2 -.-> D3[extract_tensor_transport_metadata<br>注册 NIXL 内存 / 序列化描述符]
        D3 -.-> D4[元数据回传到 owner<br>包含 nixl_serialized_descs<br>nixl_agent_meta, nixl_agent_name]
        E -.-> E1[queue_or_trigger_out_of_band_tensor_transfer]
        E1 -.-> E2[get_communicator_metadata<br>返回空的 NixlCommunicatorMetadata]
        E2 -.-> E3[只提交 __ray_recv__ 到 dst_actor<br>不提交 __ray_send__]
        E3 -.-> E4[recv_multiple_tensors<br>NIXL RDMA READ<br>直接从 src 内存读取]
        E4 -.-> E5[RDTStore.add_object]
        F -.-> F1[fetch_and_get_rdt_objects<br>use_object_store=False]
        F1 -.-> F2[fetch_multiple_tensors<br>异步发起 NIXL READ]
        F2 -.-> F3[wait_fetch_complete<br>轮询 check_xfer_state<br>直到 DONE]
    end

    style UserCode fill:#e8f5e9
    style RayInternals fill:#e3f2fd
```

---

## 5. 时序图 — 双侧传输完整流程 (NCCL/GLOO)

```mermaid
sequenceDiagram
    participant Driver as Driver (Owner)
    participant SrcActor as Source Actor
    participant DstActor as Destination Actor
    participant RDTMgr as RDTManager (Driver进程)
    participant RDTStoreSrc as RDTStore (Src进程)
    participant RDTStoreDst as RDTStore (Dst进程)
    participant Transport as TensorTransport<br>(NCCL/GLOO)
    participant Monitor as Monitor Thread

    Note over Driver: 用户调用 a1.get_tensor.remote()
    Driver->>RDTMgr: add_rdt_ref(ref, src_actor, "nccl")
    RDTMgr-->>RDTMgr: _managed_rdt_metadata[obj_id] = RDTMeta(..., tensor_transport_meta=None)

    Note over SrcActor: Actor 任务执行，返回 torch.Tensor
    SrcActor->>RDTStoreSrc: add_object_primary(obj_id, tensors, "nccl")
    RDTStoreSrc->>Transport: extract_tensor_transport_metadata(obj_id, tensors)
    Transport-->>RDTStoreSrc: CollectiveTransportMetadata(tensor_meta, tensor_device)
    RDTStoreSrc-->>RDTMgr: 元数据回传 (通过 Ray 内部机制)
    RDTMgr->>RDTMgr: set_tensor_transport_metadata_and_trigger_queued_operations(obj_id, meta)
    Note over RDTMgr: 若有 queued_transfers，立即触发

    Note over Driver: 用户调用 a2.consume.remote(ref)
    Driver->>RDTMgr: queue_or_trigger_out_of_band_tensor_transfer(dst_actor, task_args)
    RDTMgr->>Transport: get_communicator_metadata(src_actor, dst_actor, "nccl")
    Transport-->>RDTMgr: CollectiveCommunicatorMetadata(communicator_name, src_rank, dst_rank)

    RDTMgr->>SrcActor: __ray_send__(obj_id, meta, comm_meta, "nccl")<br>[concurrency_group="_ray_system"]
    RDTMgr->>DstActor: __ray_recv__(obj_id, meta, comm_meta, "nccl")<br>[concurrency_group="_ray_system"]
    RDTMgr->>Monitor: TransferMetadata(send_ref, recv_ref, ...)

    SrcActor->>RDTStoreSrc: get_object(obj_id) → tensors
    SrcActor->>Transport: send_multiple_tensors(tensors, meta, comm_meta)
    Note over Transport: collective.send(tensor, dst_rank, group_name)

    DstActor->>Transport: recv_multiple_tensors(obj_id, meta, comm_meta)
    Note over Transport: collective.recv(tensor, src_rank, group_name)
    Transport-->>DstActor: received tensors
    DstActor->>RDTStoreDst: add_object(obj_id, tensors)

    Monitor->>Monitor: ray.wait([send_ref, recv_ref])<br>检查完成/失败
    Note over Monitor: 失败时调用 _abort_transport<br>kill actors 或 destroy_collective_group

    Note over DstActor: 任务执行时反序列化
    DstActor->>RDTStoreDst: wait_and_pop_object(obj_id)
    RDTStoreDst-->>DstActor: tensors → 合入任务参数

    Note over Driver,SrcActor: ObjectRef 生命周期结束
    Driver->>RDTMgr: free_object_primary_copy / queue_or_free
    RDTMgr->>SrcActor: __ray_free__(obj_id, "nccl", meta)
    SrcActor->>Transport: garbage_collect(obj_id, meta, tensors)
```

---

## 6. 时序图 — 单侧传输完整流程 (NIXL)

```mermaid
sequenceDiagram
    participant Driver as Driver (Owner)
    participant SrcActor as Source Actor
    participant DstActor as Destination Actor
    participant RDTMgr as RDTManager (Driver进程)
    participant RDTStoreSrc as RDTStore (Src进程)
    participant RDTStoreDst as RDTStore (Dst进程)
    participant NixlTransport as NixlTensorTransport
    participant Monitor as Monitor Thread

    Note over Driver: 用户调用 a1.get_tensor.remote()
    Driver->>RDTMgr: add_rdt_ref(ref, src_actor, "nixl")

    Note over SrcActor: Actor 任务执行，返回 torch.Tensor
    SrcActor->>RDTStoreSrc: add_object_primary(obj_id, tensors, "nixl")
    SrcActor->>NixlTransport: extract_tensor_transport_metadata(obj_id, tensors)

    Note over NixlTransport: 1. cuda synchronize (GPU tensors)<br>2. 检查 memory pool eligibility<br>3. pool_eligible → _allocate_pool_xfer_descs<br>   否则 → _add_tensor_descs + get_xfer_descs<br>4. get_serialized_descs → bytes<br>5. get_agent_metadata → bytes
    NixlTransport-->>RDTStoreSrc: NixlTransportMetadata(nixl_serialized_descs, nixl_agent_meta, nixl_agent_name)
    RDTStoreSrc-->>RDTMgr: 元数据回传

    Note over Driver: 用户将 ref 传给 a2
    Driver->>RDTMgr: queue_or_trigger_out_of_band_tensor_transfer(dst_actor, task_args)
    RDTMgr->>NixlTransport: get_communicator_metadata(src, dst, "nixl")
    NixlTransport-->>RDTMgr: NixlCommunicatorMetadata() (空)

    Note over RDTMgr: is_one_sided=True → 不提交 __ray_send__
    RDTMgr->>DstActor: __ray_recv__(obj_id, meta, comm_meta, "nixl")<br>[concurrency_group="_ray_system"]

    DstActor->>NixlTransport: recv_multiple_tensors(obj_id, meta, comm_meta)
    Note over NixlTransport: 内部调用 fetch_multiple_tensors + wait_fetch_complete

    NixlTransport->>NixlTransport: fetch_multiple_tensors:
    Note over NixlTransport: 1. 创建空 tensors (或用 target_buffers)<br>2. _add_tensor_descs(tensors)<br>3. deserialize_descs → remote_xfer_descs<br>4. add_remote_agent(remote_nixl_agent_meta)<br>5. initialize_xfer("READ", local, remote, remote_name)<br>6. transfer(xfer_handle) → 返回 NixlFetchRequest

    NixlTransport->>NixlTransport: wait_fetch_complete:
    Note over NixlTransport: 轮询 check_xfer_state<br>PROC → sleep(1ms) 继续<br>DONE → 返回 tensors<br>ERR → RuntimeError

    NixlTransport-->>DstActor: received tensors
    DstActor->>RDTStoreDst: add_object(obj_id, tensors)

    RDTMgr->>Monitor: TransferMetadata(send_ref=None, recv_ref, ...)

    Note over SrcActor: ObjectRef 生命周期结束
    Driver->>RDTMgr: free_object_primary_copy
    RDTMgr->>SrcActor: __ray_free__(obj_id, "nixl", meta)
    SrcActor->>NixlTransport: garbage_collect(obj_id, meta, tensors)
    Note over NixlTransport: _managed_meta_nixl.pop(obj_id)<br>_remove_tensor_descs(tensors)<br>传统路径: deregister_memory<br>Pool路径: free_tensors → pool
```

---

## 7. 时序图 — ray.put / ray.get (NIXL 单侧)

```mermaid
sequenceDiagram
    participant User as 用户 (在 Actor 内)
    participant RDTMgr as RDTManager
    participant RDTStore as RDTStore
    participant NixlTransport as NixlTensorTransport
    participant NixlAgent as NIXL Agent

    Note over User: 在 actor 内 ray.put(tensor, _tensor_transport="nixl")
    User->>RDTMgr: put_object(obj_ref, "nixl", tensors)
    RDTMgr->>RDTStore: add_object_primary(obj_ref.hex(), tensors, "nixl")
    RDTStore->>NixlTransport: extract_tensor_transport_metadata(obj_id, tensors)
    NixlTransport->>NixlAgent: register_memory / get_xfer_descs / get_serialized_descs / get_agent_metadata
    NixlTransport-->>RDTStore: NixlTransportMetadata(...)
    RDTStore-->>RDTMgr: tensor_transport_meta (非 None，因为 ray.put 时已知)
    RDTMgr->>RDTMgr: add_rdt_ref(obj_ref, src_actor=当前actor, "nixl", meta=已知)

    Note over User: ray.get(ref) 或 传 ref 到其他 actor
    User->>RDTMgr: fetch_and_get_rdt_objects([obj_id], use_object_store=False)
    RDTMgr->>RDTMgr: _trigger_fetch(obj_id, use_object_store=False)
    Note over RDTMgr: is_one_sided("nixl")=True → 使用 tensor transport
    RDTMgr->>NixlTransport: fetch_multiple_tensors(obj_id, meta, comm_meta, target_buffers)
    NixlTransport->>NixlAgent: deserialize_descs + add_remote_agent + initialize_xfer("READ") + transfer
    NixlTransport-->>RDTMgr: NixlFetchRequest(xfer_handle, tensors, ...)
    RDTMgr->>NixlTransport: wait_fetch_complete(fetch_request, timeout)
    NixlTransport->>NixlAgent: check_xfer_state 循环直到 DONE
    NixlTransport-->>RDTMgr: tensors
    RDTMgr-->>User: 返回 tensors
```

---

## 8. 对象生命周期状态图

```mermaid
stateDiagram-v2
    [*] --> Created : ray.remote() 或 ray.put()
    Created --> MetaPending : add_rdt_ref<br>(tensor_transport_meta=None)
    MetaPending --> MetaReady : extract_tensor_transport_metadata<br>→ set_tensor_transport_metadata

    MetaPending --> QueuedTransfer : queue_or_trigger<br>(meta 未就绪时排队)
    QueuedTransfer --> Transferring : meta 就绪后触发
    MetaReady --> Transferring : trigger_out_of_band_tensor_transfer

    state Transferring {
        [*] --> SendRecv : 双侧: __ray_send__ + __ray_recv__
        [*] --> RecvOnly : 单侧: __ray_recv__ only
        SendRecv --> TransferComplete : 监控线程检查完成
        RecvOnly --> TransferComplete
    }

    Transferring --> Received : RDTStore.add_object(dst)
    Received --> Consumed : 任务反序列化<br>wait_and_pop_object

    MetaReady --> QueuedFree : free 在 meta 就绪前到达
    QueuedFree --> Freeing : meta 就绪后执行 free
    MetaReady --> Freeing : free_object_primary_copy
    Transferring --> Aborted : 监控线程检测失败/超时
    Aborted --> [*] : kill actors 或 abort_transport

    Freeing --> [*] : __ray_free__ → garbage_collect<br>_remove_tensor_descs / deregister_memory
    Consumed --> [*]

    note right of MetaPending
        ray.get 等待 meta 就绪
        超时 = RDT_FETCH_FAIL_TIMEOUT_SECONDS
    end note

    note right of Transferring
        监控线程: ray.wait + timeout 检查
        失败: abort or kill actors
    end note
```

---

## 9. NIXL 内存管理流程图

```mermaid
flowchart TB
    subgraph Registration["内存注册"]
        A[tensor 到达 extract_tensor_transport_metadata] --> B{是否有 memory_pool?}
        B -->|有 pool & 所有 tensor 在 pool 设备上<br>& 无已注册 tensor| C[pool_eligible = True]
        B -->|否| D[pool_eligible = False]
        C --> E[_allocate_pool_xfer_descs<br>从 pool 分配 MemoryBlock<br>复制 tensor 数据到 pool]
        D --> F[_add_tensor_descs<br>注册 tensor 到 NIXL<br>ref_count++]
        E --> G[_add_pool_tensor_descs<br>reg_desc=None, metadata_count=1]
        F --> H[NixlAgent.register_memory<br>返回 reg_desc]
        G --> I[NixlAgent.get_xfer_descs<br>使用 pool_tensor 的描述符]
        H --> I
        I --> J[get_serialized_descs → bytes<br>get_agent_metadata → bytes<br>构造 NixlTransportMetadata]
    end

    subgraph Transfer["传输 (RDMA READ)"]
        K[fetch_multiple_tensors] --> L[创建空 tensors<br>或使用 target_buffers]
        L --> M[_add_tensor_descs(tensors)<br>接收端注册内存]
        M --> N[deserialize_descs<br>→ remote_xfer_descs]
        N --> O[add_remote_agent<br>→ 建立远程连接]
        O --> P[initialize_xfer READ<br>local_xfer + remote_xfer]
        P --> Q[transfer → 发起 RDMA READ]
        Q --> R[返回 NixlFetchRequest]
        R --> S[wait_fetch_complete<br>轮询 check_xfer_state]
        S --> T{state?}
        T -->|DONE| U[返回 tensors]
        T -->|PROC| S
        T -->|ERR| V[Raise RuntimeError]
    end

    subgraph Cleanup["清理"]
        W[garbage_collect] --> X[_managed_meta_nixl.pop]
        X --> Y[_remove_tensor_descs]
        Y --> Z{reg_desc?}
        Z -->|有 reg_desc| AA[deregister_memory<br>NIXL 注销 + 版本号++]
        Z -->|None (pool)| AB[MemoryPoolManager.free_tensors<br>归还 MemoryBlock 到 pool]
    end

    Registration --> Transfer --> Cleanup

    style Registration fill:#fff3e0
    style Transfer fill:#e3f2fd
    style Cleanup fill:#fce4ec
```

---

## 10. 关键数据流路径总结

### 10.1 元数据流转

```
src_actor:
  task 返回 tensor → RDTStore.add_object_primary
    → TensorTransportManager.extract_tensor_transport_metadata
      → TensorTransportMetadata (含 tensor shape/dtype/device + 传输特定字段)

driver (owner):
  RDTManager.set_tensor_transport_metadata_and_trigger_queued_operations
    → 触发 queued_transfers (如有)

driver (owner):
  RDTManager.trigger_out_of_band_tensor_transfer
    → TensorTransportManager.get_communicator_metadata
      → CommunicatorMetadata
    → 提交 __ray_send__ / __ray_recv__ 到 actors (携带 meta + comm_meta)
```

### 10.2 数据流转 (双侧)

```
src_actor.__ray_send__:  RDTStore.get → send_multiple_tensors → [NCCL/GLOO send]
dst_actor.__ray_recv__:  recv_multiple_tensors → [NCCL/GLOO recv] → RDTStore.add_object
```

### 10.3 数据流转 (单侧 NIXL)

```
dst_actor.__ray_recv__:  recv_multiple_tensors → fetch_multiple_tensors
  → NIXL RDMA READ (直接读取 src 内存) → wait_fetch_complete → RDTStore.add_object
```

### 10.4 ray.get 路径

```
# NIXL (单侧):
RDTManager.fetch_and_get_rdt_objects
  → _trigger_fetch → fetch_multiple_tensors (异步 NIXL READ)
  → _wait_fetch → wait_fetch_complete (轮询直到 DONE)

# NCCL/GLOO (双侧) - 必须用 object store:
RDTManager.fetch_and_get_rdt_objects(use_object_store=True)
  → src_actor.__ray_fetch_rdt_object__
  → ray.get(object_ref) → 通过 Ray 对象存储返回 tensor
```

---

## 11. 关键源码位置索引

| 功能 | 文件 | 关键行 |
|------|------|--------|
| 公共 API 导出 | `__init__.py` | L1-29 |
| RDTManager 核心 | `rdt_manager.py` | L140-958 |
| RDTMeta 定义 | `rdt_manager.py` | L60-73 |
| TransferMetadata | `rdt_manager.py` | L77-86 |
| wait_tensor_freed | `rdt_manager.py` | L89-116 |
| set_target_for_ref | `rdt_manager.py` | L118-138 |
| add_rdt_ref | `rdt_manager.py` | L393-421 |
| trigger_out_of_band_tensor_transfer | `rdt_manager.py` | L622-742 |
| fetch_and_get_rdt_objects | `rdt_manager.py` | L775-875 |
| 监控线程 | `rdt_manager.py` | L265-391 |
| __ray_send__ / __ray_recv__ | `rdt_store.py` | L19-108 |
| __ray_free__ | `rdt_store.py` | L118-141 |
| __ray_fetch_rdt_object__ | `rdt_store.py` | L144-150 |
| RDTStore 类 | `rdt_store.py` | L163-370 |
| TensorTransportManager 抽象接口 | `tensor_transport_manager.py` | L58-295 |
| FetchRequest 基类 | `tensor_transport_manager.py` | L38-55 |
| CollectiveTensorTransport | `collective_tensor_transport.py` | L34-193 |
| NixlTensorTransport | `nixl_tensor_transport.py` | L94-666 |
| NIXL fetch (async) | `nixl_tensor_transport.py` | L279-389 |
| NIXL wait | `nixl_tensor_transport.py` | L391-447 |
| CudaIpcTransport | `cuda_ipc_transport.py` | L35-214 |
| MemoryPoolManager | `nixl_memory_pool.py` | L32-288 |
| 传输注册机制 | `util.py` | L34-216 |
| register_nixl_memory | `util.py` | L218-261 |
| register_nixl_memory_pool | `util.py` | L293-336 |