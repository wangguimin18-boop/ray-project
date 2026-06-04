.. _direct-transport:


**************************
Ray Direct Transport (RDT)
**************************

Ray Direct Transport (RDT)
Ray 直传输（RDT）

Ray objects are normally stored in Ray's CPU-based object store and copied and deserialized when accessed by a Ray task or actor.
Ray 对象通常存储在 Ray 的基于 CPU 的对象存储中，当被 Ray 任务或 Actor 访问时会被复制和反序列化。
For GPU data specifically, this can lead to unnecessary and expensive data transfers.
特别是对于 GPU 数据，这可能导致不必要且代价高昂的数据传输。
For example, passing a CUDA ``torch.Tensor`` from one Ray task to another would require a copy from GPU to CPU memory, then back again to GPU memory.
例如，将一个 CUDA ``torch.Tensor`` 从一个 Ray 任务传递到另一个 Ray 任务，需要先从 GPU 内存复制到 CPU 内存，然后再从 CPU 内存复制回 GPU 内存。

*Ray Direct Transport (RDT)* is a new feature that allows Ray to store and pass objects directly between Ray actors.
*Ray Direct Transport (RDT)* 是一项新功能，允许 Ray 在 Ray Actor 之间直接存储和传递对象。
This feature augments the familiar Ray :class:`ObjectRef <ray.ObjectRef>` API by:
该功能增强了大家熟悉的 Ray :class:`ObjectRef <ray.ObjectRef>` API，具体包括：

- Keeping GPU data in GPU memory until a transfer is necessary
- 将 GPU 数据保留在 GPU 内存中，直到需要传输时才移动
- Avoiding expensive serialization and copies to and from the Ray object store
- 避免代价高昂的序列化操作以及与 Ray 对象存储之间的来回复制
- Using efficient data transports like collective communication libraries (`Gloo <https://github.com/pytorch/gloo>`__ or `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__) or point-to-point RDMA (via `NVIDIA's NIXL <https://github.com/ai-dynamo/nixl>`__) to transfer data directly between devices, including both CPU and GPUs
- 使用高效的数据传输方式，如集合通信库（`Gloo <https://github.com/pytorch/gloo>`__ 或 `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__）或点对点 RDMA（通过 `NVIDIA 的 NIXL <https://github.com/ai-dynamo/nixl>`__），在设备之间直接传输数据，包括 CPU 和 GPU

.. note::
   RDT is currently in **alpha** and doesn't support all Ray Core APIs yet. Future releases may introduce breaking API changes. See the :ref:`limitations <limitations>` section for more details.

   RDT 目前处于 **alpha** 阶段，尚未支持所有 Ray Core API。后续版本可能会引入破坏性的 API 变更。详见 :ref:`限制 <limitations>` 章节。

Getting started
===============
入门指南

.. tip::
   RDT currently supports ``torch.Tensor`` objects created by Ray actor tasks. Other datatypes and Ray non-actor tasks may be supported in future releases.

   RDT 目前支持由 Ray Actor 任务创建的 ``torch.Tensor`` 对象。其他数据类型和 Ray 非 Actor 任务可能会在后续版本中得到支持。

This walkthrough will show how to create and use RDT with different *tensor transports*, i.e. the mechanism used to transfer the tensor between actors.
本教程将展示如何使用不同的*张量传输*（tensor transport）来创建和使用 RDT，即用于在 Actor 之间传输张量的机制。
Currently, RDT supports the following tensor transports:
目前，RDT 支持以下张量传输方式：

1. `Gloo <https://github.com/pytorch/gloo>`__: A collective communication library for PyTorch and CPUs.
1. `Gloo <https://github.com/pytorch/gloo>`__：一个用于 PyTorch 和 CPU 的集合通信库。
2. `NVIDIA NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__: A collective communication library for NVIDIA GPUs.
2. `NVIDIA NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__：一个用于 NVIDIA GPU 的集合通信库。
3. `NVIDIA NIXL <https://github.com/ai-dynamo/nixl>`__ (backed by `UCX <https://github.com/openucx/ucx>`__): A library for accelerating point-to-point transfers via RDMA, especially between various types of memory and NVIDIA GPUs.
3. `NVIDIA NIXL <https://github.com/ai-dynamo/nixl>`__（基于 `UCX <https://github.com/openucx/ucx>`__）：一个通过 RDMA 加速点对点传输的库，尤其适用于各类内存与 NVIDIA GPU 之间的传输。

For ease of following along, we'll start with the `Gloo <https://github.com/pytorch/gloo>`__ transport, which can be used without any physical GPUs.
为了便于跟随操作，我们将从 `Gloo <https://github.com/pytorch/gloo>`__ 传输方式开始，该方式无需任何物理 GPU 即可使用。

.. _direct-transport-gloo:

Usage with Gloo (CPUs only)
---------------------------
使用 Gloo（仅 CPU）

Installation
^^^^^^^^^^^
安装

.. note::
    Under construction.

    正在建设中。

Walkthrough
^^^^^^^^^^
教程

To get started, define an actor class and a task that returns a ``torch.Tensor``:
首先，定义一个 Actor 类和一个返回 ``torch.Tensor`` 的任务：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __normal_example_start__
   :end-before: __normal_example_end__

As written, when the ``torch.Tensor`` is returned, it will be copied into Ray's CPU-based object store.
按照当前的写法，当 ``torch.Tensor`` 被返回时，它会被复制到 Ray 的基于 CPU 的对象存储中。
For CPU-based tensors, this can require an expensive step to copy and serialize the object, while GPU-based tensors additionally require a copy to and from CPU memory.
对于基于 CPU 的张量，这可能需要一个代价高昂的复制和序列化步骤，而基于 GPU 的张量还需要在 CPU 内存之间来回复制。

To enable RDT, use the ``tensor_transport`` option in the :func:`@ray.method <ray.method>` decorator.
要启用 RDT，请在 :func:`@ray.method <ray.method>` 装饰器中使用 ``tensor_transport`` 选项。

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_example_start__
   :end-before: __gloo_example_end__

This decorator can be added to any actor tasks that return a ``torch.Tensor``, or that return ``torch.Tensors`` nested inside other Python objects.
此装饰器可以添加到任何返回 ``torch.Tensor`` 的 Actor 任务，或返回嵌套在其他 Python 对象中的 ``torch.Tensors`` 的 Actor 任务。
Adding this decorator will change Ray's behavior in the following ways:
添加此装饰器将以以下方式改变 Ray 的行为：

1. When returning the tensor, Ray will store a *reference* to the tensor instead of copying it to CPU memory.
1. 返回张量时，Ray 将存储该张量的*引用*，而不是将其复制到 CPU 内存。
2. When the :class:`ray.ObjectRef` is passed to another task, Ray will use Gloo to transfer the tensor to the destination task.
2. 当 :class:`ray.ObjectRef` 被传递给另一个任务时，Ray 将使用 Gloo 将张量传输到目标任务。

Note that for (2) to work, the :func:`@ray.method(tensor_transport) <ray.method>` decorator only needs to be added to the actor task that *returns* the tensor. It should not be added to actor tasks that *consume* the tensor (unless those tasks also return tensors).
注意，要让 (2) 正常工作，:func:`@ray.method(tensor_transport) <ray.method>` 装饰器只需添加到*返回*张量的 Actor 任务上。不应添加到*消费*张量的 Actor 任务上（除非这些任务也返回张量）。

Also, for (2) to work, we must first create a *collective group* of actors.
同样，要让 (2) 正常工作，我们必须先创建一个 Actor 的*集合组*（collective group）。

Creating a collective group
^^^^^^^^^^^^^^^^^^^^^^^^^^^
创建集合组

To create a collective group for use with RDT:
创建用于 RDT 的集合组：

1. Create multiple Ray actors.
1. 创建多个 Ray Actor。
2. Create a collective group on the actors using the :func:`ray.experimental.collective.create_collective_group <ray.experimental.collective.create_collective_group>` function. The `backend` specified must match the `tensor_transport` used in the :func:`@ray.method <ray.method>` decorator.
2. 使用 :func:`ray.experimental.collective.create_collective_group <ray.experimental.collective.create_collective_group>` 函数在 Actor 上创建集合组。指定的 `backend` 必须与 :func:`@ray.method <ray.method>` 装饰器中使用的 `tensor_transport` 匹配。

Here is an example:
以下是一个示例：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_group_start__
   :end-before: __gloo_group_end__

The actors can now communicate directly via gloo.
Actor 现在可以通过 Gloo 直接通信。
The group can also be destroyed using the :func:`ray.experimental.collective.destroy_collective_group <ray.experimental.collective.destroy_collective_group>` function.
该组也可以使用 :func:`ray.experimental.collective.destroy_collective_group <ray.experimental.collective.destroy_collective_group>` 函数销毁。
After calling this function, a new collective group can be created on the same actors.
调用此函数后，可以在相同的 Actor 上创建新的集合组。

Passing objects to other actors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
传递对象给其他 Actor

Now that we have a collective group, we can create and pass RDT objects between the actors.
现在我们有了集合组，可以在 Actor 之间创建和传递 RDT 对象。
Here is a full example:
以下是完整示例：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_full_example_start__
   :end-before: __gloo_full_example_end__

When the :class:`ray.ObjectRef` is passed to another task, Ray will use Gloo to transfer the tensor directly from the source actor to the destination actor instead of the default object store.
当 :class:`ray.ObjectRef` 被传递给另一个任务时，Ray 将使用 Gloo 将张量直接从源 Actor 传输到目标 Actor，而不是使用默认的对象存储。
Note that the :func:`@ray.method(tensor_transport) <ray.method>` decorator is only added to the actor task that *returns* the tensor; once this hint has been added, the receiving actor task `receiver.sum` will automatically use Gloo to receive the tensor.
注意 :func:`@ray.method(tensor_transport) <ray.method>` 装饰器仅添加到*返回*张量的 Actor 任务上；一旦添加了此提示，接收方 Actor 任务 `receiver.sum` 将自动使用 Gloo 接收张量。
In this example, because `MyActor.sum` does not have the :func:`@ray.method(tensor_transport) <ray.method>` decorator, it will use the default Ray object store transport to return `torch.sum(tensor)`.
在本示例中，由于 `MyActor.sum` 没有 :func:`@ray.method(tensor_transport) <ray.method>` 装饰器，它将使用默认的 Ray 对象存储传输来返回 `torch.sum(tensor)`。

RDT also supports passing tensors nested inside Python data structures, as well as actor tasks that return multiple tensors, like in this example:
RDT 还支持传递嵌套在 Python 数据结构中的张量，以及返回多个张量的 Actor 任务，如以下示例所示：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_multiple_tensors_example_start__
   :end-before: __gloo_multiple_tensors_example_end__

Passing RDT objects to the actor that produced them
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
将 RDT 对象传递给生成它的 Actor

RDT :class:`ray.ObjectRefs <ray.ObjectRef>` can also be passed to the actor that produced them.
RDT :class:`ray.ObjectRefs <ray.ObjectRef>` 也可以传递给生成它们的 Actor。
This avoids any copies and just provides a reference to the same ``torch.Tensor`` that was previously created.
这避免了任何复制，仅提供对先前创建的同一 ``torch.Tensor`` 的引用。
For example:
例如：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_intra_actor_start__
   :end-before: __gloo_intra_actor_end__


.. note::
    Ray only keeps a reference to the tensor created by the user, so the tensor objects are *mutable*.
    Ray 仅保留对用户创建的张量的引用，因此张量对象是*可变的*。
    If ``sender.sum`` were to modify the tensor in the above example, the changes would also be seen by ``receiver.sum``.
    如果在上面的示例中 ``sender.sum`` 修改了张量，那么 ``receiver.sum`` 也会看到这些修改。
    This differs from the normal Ray Core API, which always makes an immutable copy of data returned by actors.
    这与正常的 Ray Core API 不同，后者总是对 Actor 返回的数据创建一个不可变的副本。


``ray.get``
^^^^^^^^^^^
``ray.get``

The :func:`ray.get <ray.get>` function can also be used as usual to retrieve the result of an RDT object. However, :func:`ray.get <ray.get>` will by default use the same tensor transport as the one specified in the :func:`@ray.method <ray.method>` decorator. For collective-based transports, this will not work if the caller is not part of the collective group.
:func:`ray.get <ray.get>` 函数也可以像往常一样用于获取 RDT 对象的结果。然而，:func:`ray.get <ray.get>` 默认将使用与 :func:`@ray.method <ray.method>` 装饰器中指定的相同的张量传输方式。对于基于集合的传输方式，如果调用者不是集合组的成员，这将无法工作。

Therefore, users need to specify the Ray object store as the tensor transport explicitly by setting ``_use_object_store`` in :func:`ray.get <ray.get>`.
因此，用户需要在 :func:`ray.get <ray.get>` 中通过设置 ``_use_object_store`` 来显式指定 Ray 对象存储作为张量传输方式。

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_get_start__
   :end-before: __gloo_get_end__

Object mutability
^^^^^^^^^^^^^^^^
对象可变性

Unlike objects in the Ray object store, RDT objects are *mutable*, meaning that Ray only holds a reference to the tensor and will not copy it until a transfer is requested.
与 Ray 对象存储中的对象不同，RDT 对象是*可变的*，这意味着 Ray 仅持有对张量的引用，直到请求传输时才会复制。
This means that if the actor that returns a tensor also keeps a reference to the tensor, and the actor later modifies it in place while Ray is still storing the tensor reference, it's possible that some or all of the changes may be seen by receiving actors.
这意味着如果返回张量的 Actor 同时保留了对该张量的引用，并且该 Actor 后来在 Ray 仍然存储该张量引用时就地修改了它，接收方 Actor 可能会看到部分或全部修改。

Here is an example of what can go wrong:
以下是可能出现问题的示例：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_wait_tensor_freed_bad_start__
   :end-before: __gloo_wait_tensor_freed_bad_end__

In this example, the sender actor returns a tensor to Ray, but it also keeps a reference to the tensor in its local state.
在本示例中，发送方 Actor 将张量返回给 Ray，但它也在本地状态中保留了对该张量的引用。
Then, in `sender.increment_and_sum_stored_tensor`, the sender actor modifies the tensor in place while Ray is still holding the tensor reference.
然后，在 `sender.increment_and_sum_stored_tensor` 中，发送方 Actor 在 Ray 仍然持有张量引用时就地修改了该张量。
Then, the `receiver.increment_and_sum` task receives the modified tensor instead of the original, so the assertion fails.
接着，`receiver.increment_and_sum` 任务接收到的是修改后的张量而非原始张量，因此断言失败。

To fix this kind of error, use the :func:`ray.experimental.wait_tensor_freed <ray.experimental.wait_tensor_freed>` function to wait for Ray to release all references to the tensor, so that the actor can safely write to the tensor again.
要修复此类错误，请使用 :func:`ray.experimental.wait_tensor_freed <ray.experimental.wait_tensor_freed>` 函数等待 Ray 释放对张量的所有引用，以便 Actor 可以安全地再次写入该张量。
:func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` will unblock once all tasks that depend on the tensor have finished executing and all corresponding `ObjectRefs` have gone out of scope.
:func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` 将在所有依赖该张量的任务执行完毕且所有对应的 `ObjectRef` 超出作用域后解除阻塞。
Ray tracks tasks that depend on the tensor by keeping track of which tasks take the `ObjectRef` corresponding to the tensor as an argument.
Ray 通过跟踪哪些任务将对应张量的 `ObjectRef` 作为参数来追踪依赖该张量的任务。

Here's a fixed version of the earlier example.
以下是之前示例的修复版本。

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_wait_tensor_freed_start__
   :end-before: __gloo_wait_tensor_freed_end__

The main changes are:
主要修改如下：
1. `sender` calls :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` before modifying the tensor in place.
1. `sender` 在就地修改张量之前调用 :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>`。
2. The driver skips :func:`ray.get <ray.get>` because :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` blocks until all `ObjectRefs` pointing to the tensor are freed, so calling :func:`ray.get <ray.get>` here would cause a deadlock.
2. 驱动程序跳过 :func:`ray.get <ray.get>`，因为 :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` 会阻塞直到所有指向该张量的 `ObjectRef` 被释放，因此在此处调用 :func:`ray.get <ray.get>` 会导致死锁。
3. The driver calls `del tensor` to release its reference to the tensor. Again, this is necessary because :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` blocks until all `ObjectRefs` pointing to the tensor are freed.
3. 驱动程序调用 `del tensor` 以释放其对张量的引用。同样，这是必要的，因为 :func:`wait_tensor_freed <ray.experimental.wait_tensor_freed>` 会阻塞直到所有指向该张量的 `ObjectRef` 被释放。

When an RDT `ObjectRef` is passed back to the same actor that produced it, Ray passes back a *reference* to the tensor instead of a copy. Therefore, the same kind of bug can occur.
当 RDT `ObjectRef` 被传递回生成它的同一 Actor 时，Ray 会传递回张量的*引用*而非副本。因此，也可能发生相同类型的错误。
To help catch such cases, Ray will print a warning if an RDT object is passed to the actor that produced it and a different actor, like so:
为了帮助捕获此类情况，如果 RDT 对象被传递给生成它的 Actor 和另一个不同的 Actor，Ray 将打印警告，如下所示：

.. literalinclude:: ../doc_code/direct_transport_gloo.py
   :language: python
   :start-after: __gloo_object_mutability_warning_start__
   :end-before: __gloo_object_mutability_warning_end__


Usage with NCCL (NVIDIA GPUs only)
----------------------------------
使用 NCCL（仅 NVIDIA GPU）

RDT requires just a few lines of code change to switch tensor transports. Here is the :ref:`Gloo example <direct-transport-gloo>`, modified to use NVIDIA GPUs and the `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__ library for collective GPU communication.
RDT 只需更改几行代码即可切换张量传输方式。以下是 :ref:`Gloo 示例 <direct-transport-gloo>`，修改为使用 NVIDIA GPU 和 `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__ 库进行 GPU 集合通信。

.. literalinclude:: ../doc_code/direct_transport_nccl.py
   :language: python
   :start-after: __nccl_full_example_start__
   :end-before: __nccl_full_example_end__

The main code differences are:
主要代码差异如下：

1. The :func:`@ray.method <ray.method>` uses ``tensor_transport="nccl"`` instead of ``tensor_transport="gloo"``.
1. :func:`@ray.method <ray.method>` 使用 ``tensor_transport="nccl"`` 而非 ``tensor_transport="gloo"``。
2. The :func:`ray.experimental.collective.create_collective_group <ray.experimental.collective.create_collective_group>` function is used to create a collective group.
2. 使用 :func:`ray.experimental.collective.create_collective_group <ray.experimental.collective.create_collective_group>` 函数创建集合组。
3. The tensor is created on the GPU using the ``.cuda()`` method.
3. 使用 ``.cuda()`` 方法在 GPU 上创建张量。

Usage with NIXL (CPUs or NVIDIA GPUs)
-------------------------------------
使用 NIXL（CPU 或 NVIDIA GPU）

Installation
^^^^^^^^^^^
安装

First, install NIXL with a plain ``pip install nixl``.
首先，使用简单的 ``pip install nixl`` 安装 NIXL。
For maximum performance, run the `install_gdrcopy.sh <https://github.com/ray-project/ray/blob/master/doc/tools/install_gdrcopy.sh>`__ script (e.g., ``install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"``). You can find available OS versions `here <https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/>`__. 
为获得最大性能，请运行 `install_gdrcopy.sh <https://github.com/ray-project/ray/blob/master/doc/tools/install_gdrcopy.sh>`__ 脚本（例如 ``install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"``）。可在 `此处 <https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/>`__ 查找可用的操作系统版本。

Note that you should also set these UCX environment variables to either let UCX choose the right transport from all options, or so that you can yourself set your preferred transport option.
请注意，您还应设置这些 UCX 环境变量，以便让 UCX 从所有选项中选择合适的传输方式，或者让您自行设置首选的传输选项。


.. code-block:: bash

   # Example UCX configuration, adjust according to your environment
   # UCX 配置示例，请根据您的环境进行调整
   $ export UCX_TLS=all  # or specify specific transports like "rc,ud,sm,^cuda_ipc" ..etc
   # 或指定特定的传输方式，如 "rc,ud,sm,^cuda_ipc" 等
   $ export UCX_NET_DEVICES=all  # or specify network devices like "mlx5_0:1,mlx5_1:1"
   # 或指定网络设备，如 "mlx5_0:1,mlx5_1:1"


Walkthrough
^^^^^^^^^^
教程

NIXL can transfer data between different devices, including CPUs and NVIDIA GPUs, but doesn't require a collective group to be created ahead of time.
NIXL 可以在不同设备之间传输数据，包括 CPU 和 NVIDIA GPU，但不需要预先创建集合组。
This means that any actor that has NIXL installed in its environment can be used to create and pass an RDT object.
这意味着环境中安装了 NIXL 的任何 Actor 都可以用于创建和传递 RDT 对象。

Otherwise, the usage is the same as in the :ref:`Gloo example <direct-transport-gloo>`.
除此之外，用法与 :ref:`Gloo 示例 <direct-transport-gloo>` 相同。

Here is an example showing how to use NIXL to transfer an RDT object between two actors:
以下是一个展示如何使用 NIXL 在两个 Actor 之间传输 RDT 对象的示例：

.. literalinclude:: ../doc_code/direct_transport_nixl.py
   :language: python
   :start-after: __nixl_full_example_start__
   :end-before: __nixl_full_example_end__

Compared to the :ref:`Gloo example <direct-transport-gloo>`, the main code differences are:
与 :ref:`Gloo 示例 <direct-transport-gloo>` 相比，主要代码差异如下：

1. The :func:`@ray.method <ray.method>` uses ``tensor_transport="nixl"`` instead of ``tensor_transport="gloo"``.
1. :func:`@ray.method <ray.method>` 使用 ``tensor_transport="nixl"`` 而非 ``tensor_transport="gloo"``。
2. No collective group is needed.
2. 不需要集合组。

ray.put and ray.get with NIXL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NIXL 的 ray.put 和 ray.get

Unlike the collective-based tensor transports (Gloo and NCCL), the :func:`ray.get <ray.get>` function can use NIXL to retrieve a copy of the result.
与基于集合的张量传输方式（Gloo 和 NCCL）不同，:func:`ray.get <ray.get>` 函数可以使用 NIXL 来获取结果的副本。
By default, the tensor transport for :func:`ray.get <ray.get>` will be the one specified in the :func:`@ray.method <ray.method>` decorator.
默认情况下，:func:`ray.get <ray.get>` 的张量传输方式将是 :func:`@ray.method <ray.method>` 装饰器中指定的方式。

.. literalinclude:: ../doc_code/direct_transport_nixl.py
   :language: python
   :start-after: __nixl_get_start__
   :end-before: __nixl_get_end__

You can also use NIXL to retrieve the result from references created by :func:`ray.put <ray.put>`.
您也可以使用 NIXL 从 :func:`ray.put <ray.put>` 创建的引用中获取结果。

.. literalinclude:: ../doc_code/direct_transport_nixl.py
   :language: python
   :start-after: __nixl_put__and_get_start__
   :end-before: __nixl_put__and_get_end__


Summary
-------
总结

RDT allows Ray to store and pass objects directly between Ray actors, using accelerated transports like GLOO, NCCL, and NIXL.
RDT 允许 Ray 使用加速传输方式（如 GLOO、NCCL 和 NIXL）在 Ray Actor 之间直接存储和传递对象。
Here are the main points to keep in mind:
以下是需要记住的主要要点：

* If using a collective-based tensor transport (Gloo or NCCL), a collective group must be created ahead of time. NIXL just requires all involved actors to have NIXL installed.
* 如果使用基于集合的张量传输方式（Gloo 或 NCCL），必须预先创建集合组。NIXL 仅要求所有涉及的 Actor 已安装 NIXL。
* Unlike objects in the Ray object store, RDT objects are *mutable*, meaning that Ray only holds a reference, not a copy, to the stored tensor(s). 
* 与 Ray 对象存储中的对象不同，RDT 对象是*可变的*，这意味着 Ray 仅持有对存储张量的引用，而非副本。
* Otherwise, actors can be used as normal.
* 除此之外，Actor 可以正常使用。

For a full list of limitations, see the :ref:`limitations <limitations>` section.
有关完整的限制列表，请参见 :ref:`限制 <limitations>` 章节。


Microbenchmarks
===============
微基准测试

.. note::
    Under construction.

    正在建设中。

.. _limitations:

Limitations
===========
限制

RDT is currently in alpha and currently has the following limitations, which may be addressed in future releases:
RDT 目前处于 alpha 阶段，当前有以下限制，这些限制可能会在后续版本中得到解决：

* Support for ``torch.Tensor`` objects only.
* 仅支持 ``torch.Tensor`` 对象。
* Support for Ray actors only, not Ray tasks.
* 仅支持 Ray Actor，不支持 Ray 任务。
* Support for the following transports: GLOO, NCCL, and NIXL.
* 支持以下传输方式：GLOO、NCCL 和 NIXL。
* Support for CPUs and NVIDIA GPUs only.
* 仅支持 CPU 和 NVIDIA GPU。
* RDT objects are *mutable*. This means that Ray only holds a reference to the tensor, and will not copy it until a transfer is requested. Thus, if the application code also keeps a reference to a tensor before returning it, and modifies the tensor in place, then some or all of the changes may be seen by the receiving actor.
* RDT 对象是*可变的*。这意味着 Ray 仅持有对张量的引用，直到请求传输时才会复制。因此，如果应用程序代码在返回张量之前也保留了对张量的引用，并就地修改了该张量，那么接收方 Actor 可能会看到部分或全部修改。
* `await` on an RDT ref is temporarily not supported.
* 暂不支持对 RDT 引用使用 `await`。

For collective-based / two-sided tensor transports (Gloo and NCCL):
对于基于集合/双侧的张量传输方式（Gloo 和 NCCL）：

* Only the process that created the collective group can submit actor tasks that return and pass RDT objects. If the creating process passes the actor handles to other processes, those processes can submit actor tasks as usual, but will not be able to use RDT objects.
* 只有创建集合组的进程才能提交返回和传递 RDT 对象的 Actor 任务。如果创建进程将 Actor 句柄传递给其他进程，这些进程可以照常提交 Actor 任务，但无法使用 RDT 对象。
* Similarly, the process that created the collective group cannot serialize and pass RDT :class:`ray.ObjectRefs <ray.ObjectRef>` to other Ray tasks or actors. Instead, the :class:`ray.ObjectRef`\s can only be passed as direct arguments to other actor tasks, and those actors must be in the same collective group.
* 同样，创建集合组的进程不能序列化并将 RDT :class:`ray.ObjectRefs <ray.ObjectRef>` 传递给其他 Ray 任务或 Actor。相反，:class:`ray.ObjectRef`\s 只能作为直接参数传递给其他 Actor 任务，且这些 Actor 必须在同一集合组中。
* Each actor can only be in one collective group per tensor transport at a time.
* 每个 Actor 在同一时间每种张量传输方式只能属于一个集合组。
* No support for :func:`ray.put <ray.put>`.
* 不支持 :func:`ray.put <ray.put>`。
* No support for out-of-order actors such as async actors or actors with ``max_concurrency`` > 1.
* 不支持乱序 Actor，如异步 Actor 或 ``max_concurrency`` > 1 的 Actor。


Due to a known issue, for NIXL, we currently do not support storing different GPU objects at the same actor, where the objects contain an overlapping but not equal set of tensors. To support this pattern, ensure that the first `ObjectRef` has gone out of scope before storing the same tensor(s) again in a second object.
由于一个已知问题，对于 NIXL，我们目前不支持在同一 Actor 上存储不同的 GPU 对象，其中这些对象包含重叠但不完全相同的张量集合。要支持此模式，请确保第一个 `ObjectRef` 已超出作用域后再将相同的张量再次存储到第二个对象中。

.. literalinclude:: ../doc_code/direct_transport_nixl.py
   :language: python
   :start-after: __nixl_limitations_start__
   :end-before: __nixl_limitations_end__

Error handling
=============
错误处理

* Application-level errors, i.e. exceptions raised by user code, will not destroy the collective group and will instead be propagated to any dependent task(s), as for non-RDT Ray objects.
* 应用层面的错误，即用户代码抛出的异常，不会销毁集合组，而是像非 RDT Ray 对象一样传播到依赖的任务。

* If a system-level error occurs during a GLOO or NCCL collective operation, the collective group will be destroyed and the actors will be killed to prevent any hanging.
* 如果在 GLOO 或 NCCL 集合操作过程中发生系统层面的错误，集合组将被销毁且 Actor 将被终止，以防止任何挂起。

* If a system-level error occurs during a NIXL transfer, Ray or NIXL will abort the transfer with an exception and Ray will raise the exception in the dependent task or on the ray.get on the NIXL ref.
* 如果在 NIXL 传输过程中发生系统层面的错误，Ray 或 NIXL 将以异常中止传输，Ray 将在依赖任务或 NIXL 引用的 ray.get 上抛出该异常。

* System-level errors include:
* 系统层面的错误包括：
   * Errors internal to the third-party transport, e.g., NCCL network errors
   * 第三方传输内部的错误，例如 NCCL 网络错误
   * Actor or node failures
   * Actor 或节点故障
   * Transport errors due to tensor device / transport mismatches, e.g., a CPU tensor when using NCCL
   * 由于张量设备/传输方式不匹配导致的传输错误，例如使用 NCCL 时出现了 CPU 张量
   * Ray RDT object fetch timeouts (can be overridden by setting the ``RAY_rdt_fetch_fail_timeout_milliseconds`` environment variable)
   * Ray RDT 对象获取超时（可通过设置 ``RAY_rdt_fetch_fail_timeout_milliseconds`` 环境变量来覆盖）
   * Any unexpected system bugs
   * 任何意外的系统缺陷


Advanced: Registering a custom tensor transport
===============================================
高级：注册自定义张量传输

Ray allows you to register custom tensor transports at runtime for use with RDT.
Ray 允许您在运行时注册自定义张量传输方式以供 RDT 使用。
To implement a custom tensor transport, you can implement the abstract interface :class:`ray.experimental.TensorTransportManager <ray.experimental.TensorTransportManager>` and register it using :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>`.
要实现自定义张量传输，您可以实现抽象接口 :class:`ray.experimental.TensorTransportManager <ray.experimental.TensorTransportManager>` 并使用 :func:`ray.experimental.register_tensor_transport <ray.experimental.register_tensor_transport>` 注册。

For a complete guide on implementing custom tensor transports, including detailed documentation of all required methods, see :ref:`custom-tensor-transport`.
有关实现自定义张量传输的完整指南，包括所有必需方法的详细文档，请参见 :ref:`custom-tensor-transport`。


Advanced: RDT Internals
=======================
高级：RDT 内部机制

.. note::
    Under construction.

    正在建设中。

Table of Contents
-----------------
目录

Learn more details about Ray Direct Transport from the following links.
通过以下链接了解更多 Ray Direct Transport 的详细信息。

.. toctree::
    :maxdepth: 1

    custom-tensor-transport
