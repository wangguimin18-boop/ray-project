.. _custom-tensor-transport:

*************************************************
Implementing a custom tensor transport (Advanced)
实现自定义 tensor 传输（高级）
*************************************************

Ray Direct Transport (RDT) allows you to register custom tensor transports at runtime.
Ray Direct Transport (RDT) 允许你在运行时注册自定义 tensor 传输。
This page explains how to implement a custom tensor transport by implementing the :class:`TensorTransportManager <ray.experimental.TensorTransportManager>` abstract interface.
本页说明如何通过实现 :class:`TensorTransportManager <ray.experimental.TensorTransportManager>` 抽象接口来创建自定义 tensor 传输。

Overview
概述
========

To create a custom tensor transport:
创建自定义 tensor 传输的步骤：

1. Implement the abstract interface :class:`ray.experimental.TensorTransportManager <ray.experimental.TensorTransportManager>`.
1. 实现抽象接口 :class:`ray.experimental.TensorTransportManager <ray.experimental.TensorTransportManager>`。
2. Define custom metadata classes by extending :class:`TensorTransportMetadata <ray.experimental.TensorTransportMetadata>` and :class:`CommunicatorMetadata <ray.experimental.CommunicatorMetadata>`.
2. 通过扩展 :class:`TensorTransportMetadata <ray.experimental.TensorTransportMetadata>` 和 :class:`CommunicatorMetadata <ray.experimental.CommunicatorMetadata>` 来定义自定义元数据类。
3. Register your transport using :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>`.
3. 使用 :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>` 注册你的传输。

When Ray needs to transfer a tensor between actors using your transport, it calls specific methods on your ``TensorTransportManager`` implementation at different stages of the transfer lifecycle.
当 Ray 需要使用你的传输在 actor 之间传输 tensor 时，它会在传输生命周期的不同阶段调用你的 ``TensorTransportManager`` 实现上的特定方法。


Implementing TensorTransportManager
实现 TensorTransportManager
===================================

The :class:`TensorTransportManager <ray.experimental.TensorTransportManager>` abstract class defines the interface for custom tensor transports. You must implement all abstract methods.
:class:`TensorTransportManager <ray.experimental.TensorTransportManager>` 抽象类定义了自定义 tensor 传输的接口。你必须实现所有抽象方法。

The following diagram shows when each method is called during a tensor transfer:
下图展示了在 tensor 传输过程中每个方法何时被调用：

.. code-block:: text

   Source Actor                    Owner Process                 Destination Actor
   源 Actor                         所有者进程                      目标 Actor
   ============                    =============                 =================
        |                               |                               |
   1. Task returns tensor               |                               |
      ``extract_tensor_transport_metadata``                             |
      1. 任务返回 tensor                 |                               |
      ``extract_tensor_transport_metadata``                             |
        |                               |                               |
        | ---- transport_metadata ----> |                               |
        |                               |                               |
        |                     2. Prepare communicator                   |
        |                        ``get_communicator_metadata``          |
        |                     2. 准备通信器                                |
        |                        ``get_communicator_metadata``          |
        |                               |                               |
        | <---- comm metadata --------- | ---- comm metadata -------->  |
        |                               |                               |
   3. ``send_multiple_tensors``         |          3. ``recv_multiple_tensors``
                                        |                               |
        | ------------ tensors ---------------------------------------> |
        |                               |                               |
|                         (transfer complete)                   |
         |                         (传输完成)                             |
         |                               |                               |
         |                      5. Ref goes out of scope                 |
         |                      5. 引用超出作用域                          |
         | <---------------------------- |                               |
    5. Clean up resources                |                               |
    5. 清理资源                           |                               |
       ``garbage_collect``               |                               |


Note that Ray will not call `send_multiple_tensors` for one-sided transports.
注意，对于单侧传输（one-sided transports），Ray 不会调用 `send_multiple_tensors`。
The following diagram shows where each method is called in the ray.put / ray.get case supported by one-sided transports.
下图展示了在单侧传输支持的 ray.put / ray.get 场景中每个方法的调用位置。

.. code-block:: text

Source Actor                                                  Destination Actor
   源 Actor                                                      目标 Actor
   ============                                                  =================
        |                                                               |
1. User ``ray.put``'s tensor                                         |
       ``extract_tensor_transport_metadata``                             |
       1. 用户 ``ray.put`` 的 tensor                                    |
        |                                                               |
        |                                                               |
   2. User passes ref to another actor                                  |
       2. 用户将 ref 传递给另一个 actor                                   |
        | ---- transport_metadata ---------------------------------->   |
        |                                                               |
        |                                                               |
        |                                          3. User ``ray.get``'s on object ref
                                                      3. 用户对 object ref 执行 ``ray.get``
                                                       ``get_communicator_metadata``
        |                                              ``recv_multiple_tensors``
        | ------------ tensors --------- -----------------------------> |
        |                                                               |
        |                         (transfer complete)                   |
        |                         (传输完成)                             |
        |                                                               |
   4. Clean up resources                                                |
   4. 清理资源                                                          |
      ``garbage_collect``                                               |
   (when ref goes out of scope)                                         |
   (当引用超出作用域时)                                                   |


The API reference page for :class:`TensorTransportManager <ray.experimental.TensorTransportManager>` has more details on what each method does and how to implement them.
:class:`TensorTransportManager <ray.experimental.TensorTransportManager>` 的 API 参考页有更多关于每个方法的功能和实现方式的详细信息。
See implementations of Ray's default transports (NCCL, NIXL, etc.) in the `python/ray/experimental/rdt/ <https://github.com/ray-project/ray/tree/master/python/ray/experimental/rdt>`_ directory.
可以在 `python/ray/experimental/rdt/ <https://github.com/ray-project/ray/tree/master/python/ray/experimental/rdt>`_ 目录中查看 Ray 默认传输（NCCL、NIXL 等）的实现。
The following is an walk-through for implementing and using a custom tensor transport.
以下是一个实现和使用自定义 tensor 传输的完整 walkthrough。

Example: Shared memory tensor transport
示例：共享内存 tensor 传输
========================================

The following walks through a complete custom tensor transport that transfers ``numpy`` arrays through shared memory.
以下 walkthrough 展示了一个通过共享内存传输 ``numpy`` 数组的完整自定义 tensor 传输。

Note that because shared memory is one-sided (the receiver directly reads the memory block the sender wrote to),
注意，因为共享内存是单侧的（接收方直接读取发送方写入的内存块），
``is_one_sided`` returns ``True`` and Ray never calls ``send_multiple_tensors``.
``is_one_sided`` 返回 ``True``，且 Ray 不会调用 ``send_multiple_tensors``。

Define metadata classes
定义元数据类
-----------------------

Your transport uses two metadata classes that flow through different stages of the transfer:
你的传输使用两个元数据类，它们在传输的不同阶段流转：

- :class:`TensorTransportMetadata <ray.experimental.TensorTransportMetadata>` is created on the **source actor** during ``extract_tensor_transport_metadata``. It carries per-tensor information (shapes, dtypes, devices) plus any transport-specific identifiers (e.g., shared memory block names, RDMA keys) that the receiver needs to locate and read the data.
- :class:`TensorTransportMetadata <ray.experimental.TensorTransportMetadata>` 在 **源 actor** 上于 ``extract_tensor_transport_metadata`` 期间创建。它携带每个 tensor 的信息（形状、数据类型、设备）以及接收方定位和读取数据所需的任何传输特定标识符（例如共享内存块名称、RDMA 密钥）。

- :class:`CommunicatorMetadata <ray.experimental.CommunicatorMetadata>` is created on the **owner/driver process** during ``get_communicator_metadata``. It carries any coordination information both actors need, such as ranks in a collective group. For one-sided transports (where the receiver can directly read the sender's memory), an empty metadata object is typically sufficient.
- :class:`CommunicatorMetadata <ray.experimental.CommunicatorMetadata>` 在 **所有者/驱动进程** 上于 ``get_communicator_metadata`` 期间创建。它携带两个 actor 都需要的协调信息，例如集合通信组中的 rank。对于单侧传输（接收方可以直接读取发送方的内存），通常一个空的元数据对象就够了。

Start by extending these classes to carry any transport-specific state.
首先扩展这些类以携带任何传输特定的状态。
``ShmTransportMetadata`` stores the shared memory block name and size so the receiver can locate and read the data.
``ShmTransportMetadata`` 存储共享内存块的名称和大小，以便接收方定位和读取数据。
This transport doesn't need any communicator metadata, so ``ShmCommunicatorMetadata`` is empty.
此传输不需要任何通信器元数据，因此 ``ShmCommunicatorMetadata`` 为空。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_metadata_start__
   :end-before: __custom_metadata_end__

Extract tensor transport metadata
提取 tensor 传输元数据
---------------------------------

Ray calls ``extract_tensor_transport_metadata`` on the source actor right after the task produces its result tensors.
Ray 在任务产生结果 tensor 后立即在源 actor 上调用 ``extract_tensor_transport_metadata``。
Record shapes and dtypes, then perform any transport-specific registration. Here, the implementation serializes the tensors
记录形状和数据类型，然后执行任何传输特定的注册。在此实现中，将 tensor 序列化
into a shared memory block and records the block name and size in the metadata so the receiver can find it.
到共享内存块中，并在元数据中记录块名称和大小，以便接收方找到它。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_extract_start__
   :end-before: __custom_extract_end__

Get communicator metadata
获取通信器元数据
-------------------------

Ray calls ``get_communicator_metadata`` on the owner/driver process before orchestrating the transfer.
Ray 在协调传输之前，在所有者/驱动进程上调用 ``get_communicator_metadata``。
Return any information both actors need to coordinate, such as ranks in a collective group.
返回两个 actor 协调所需的任何信息，例如集合通信组中的 rank。
For one-sided transports such as shared memory, an empty metadata object is fine.
对于像共享内存这样的单侧传输，空的元数据对象即可。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_communicator_start__
   :end-before: __custom_communicator_end__

Transport properties
传输属性
--------------------

Define your ``TensorTransportManager`` subclass and implement the property methods.
定义你的 ``TensorTransportManager`` 子类并实现属性方法。
``tensor_transport_backend`` returns the name that users pass to ``@ray.method(tensor_transport=...)``.
``tensor_transport_backend`` 返回用户传递给 ``@ray.method(tensor_transport=...)`` 的名称。
``is_one_sided`` and ``can_abort_transport`` tell Ray how to orchestrate transfers and handle errors.
``is_one_sided`` 和 ``can_abort_transport`` 告知 Ray 如何协调传输和处理错误。
``actor_has_tensor_transport`` lets Ray check whether a given actor can use this transport.
``actor_has_tensor_transport`` 让 Ray 检查给定 actor 是否可以使用此传输。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_properties_start__
   :end-before: __custom_properties_end__

Send and receive
发送与接收
----------------

``recv_multiple_tensors`` runs on the destination actor. For this shared memory transport, it opens the
``recv_multiple_tensors`` 在目标 actor 上运行。对于此共享内存传输，它按名称打开
shared memory block by name and deserializes the tensors.
共享内存块并反序列化 tensor。

``send_multiple_tensors`` runs on the source actor for two-sided transports. Since shared memory is one-sided,
``send_multiple_tensors`` 在源 actor 上运行，用于双侧传输。由于共享内存是单侧的，
Ray never calls this method, so it raises ``NotImplementedError`` as a safety guard.
Ray 不会调用此方法，因此它抛出 ``NotImplementedError`` 作为安全保护。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_send_recv_start__
   :end-before: __custom_send_recv_end__

Cleanup
清理
-------

``garbage_collect`` runs on the source actor when Ray's reference counting determines the object is out of scope.
``garbage_collect`` 在 Ray 的引用计数确定对象已超出作用域时，在源 actor 上运行。
Release any transport resources here, in this case closing and unlinking the shared memory block.
在此释放任何传输资源，本例中为关闭和取消链接共享内存块。

``abort_transport`` runs on both actors when a system error occurs during transfer, if ``can_abort_transport`` returns ``True``.
``abort_transport`` 在传输期间发生系统错误时，如果 ``can_abort_transport`` 返回 ``True``，则在两个 actor 上运行。
Since this transport returns ``False`` for ``can_abort_transport``, Ray kills the involved actors instead,
由于此传输的 ``can_abort_transport`` 返回 ``False``，Ray 会终止涉及的 actor，
so ``abort_transport`` is a no-op.
因此 ``abort_transport`` 是空操作（no-op）。

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_cleanup_start__
   :end-before: __custom_cleanup_end__

Registering your transport
注册你的传输
==========================

After implementing your transport, the **driver process** must register it with :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>` before creating any actors that use it:
实现传输后，**驱动进程** 必须在创建使用它的任何 actor 之前，通过 :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>` 注册它：

.. literalinclude:: ../doc_code/direct_transport_custom.py
   :language: python
   :start-after: __custom_usage_start__
   :end-before: __custom_usage_end__


Limitations
限制
===========

Custom tensor transports have the following limitations:
自定义 tensor 传输有以下限制：

- **Actor restarts aren't supported.** Your actor doesn't have access to the custom transport after a restart.
- **不支持 Actor 重启。** Actor 重启后无法访问自定义传输。

- **Register transports before actor creation.** If you register a transport after creating an actor, that actor can't use the new transport.
- **在创建 actor 之前注册传输。** 如果在创建 actor 之后注册传输，该 actor 无法使用新传输。

- **Out-of-order actors** If you have an out-of-order actor (such as an async actor) and the process where you submit the actor task is different from where you created the actor, Ray can't guarantee it has registered your custom transport on the actor at task execution time.
- **乱序 actor** 如果你有一个乱序 actor（如异步 actor），且提交 actor 任务的进程与创建 actor 的进程不同，Ray 无法保证在任务执行时已在 actor 上注册了你的自定义传输。

- **Actor creation and task submission from different processes** If the process where you submit an actor task is different from where you created the actor, Ray can't guarantee it has registered your custom transport on the actor at task execution time.
- **从不同进程创建 actor 和提交任务** 如果提交 actor 任务的进程与创建 actor 的进程不同，Ray 无法保证在任务执行时已在 actor 上注册了你的自定义传输。

For general RDT limitations, see :ref:`limitations <limitations>`.
关于 RDT 的通用限制，请参见 :ref:`limitations <limitations>`。

Also feel free to reach out through `GitHub issues <https://github.com/ray-project/ray/issues>`_ or the `Ray Slack <https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform>`_ to ask any questions.
如有任何问题，也可以通过 `GitHub issues <https://github.com/ray-project/ray/issues>`_ 或 `Ray Slack <https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform>`_ 联系我们。
