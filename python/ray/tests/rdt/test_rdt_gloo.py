import logging
import random
import re
import sys
import threading
import time

import pytest
import torch

import ray
from ray._common.test_utils import SignalActor, wait_for_condition
from ray.experimental.collective import create_collective_group
from ray.experimental.rdt.collective_tensor_transport import (
    CollectiveTransportMetadata,
)

# tensordict is not supported on macos ci, so we skip the tests
# tensordict 在 macOS CI 上不受支持，因此跳过这些测试
support_tensordict = sys.platform != "darwin"

if support_tensordict:
    from tensordict import TensorDict


# TODO: check whether concurrency groups are created correctly if
# enable_tensor_transport is True or if any methods are decorated with
# @ray.method(tensor_transport=...). Check that specifying
# .options(tensor_transport=...) fails if enable_tensor_transport is False.
# TODO: 检查当 enable_tensor_transport 为 True 或任何方法被
# @ray.method(tensor_transport=...) 装饰时，并发组是否正确创建。
# 检查当 enable_tensor_transport 为 False 时，指定
# .options(tensor_transport=...) 是否会失败。
@ray.remote
class GPUTestActor:
    @ray.method(tensor_transport="gloo")
    def echo(self, data):
        return data

    def add(self, a, b):
        return a + b

    def double(self, data):
        if isinstance(data, list):
            return [self.double(d) for d in data]
        if support_tensordict and isinstance(data, TensorDict):
            return data.apply(lambda x: x * 2)
        return data * 2

    def increment(self, data):
        data += 1
        return data

    def get_out_of_band_tensors(self, obj_id: str, timeout=None):
        rdt_store = ray._private.worker.global_worker.rdt_manager.rdt_store
        if timeout is None:
            timeout = 0
        return rdt_store.wait_and_get_object(obj_id, timeout)

    def get_num_rdt_objects(self):
        rdt_manager = ray._private.worker.global_worker.rdt_manager
        return rdt_manager.rdt_store.get_num_objects()

    def fail(self, error_message):
        raise Exception(error_message)


@ray.remote
class ErrorActor:
    @ray.method(tensor_transport="gloo")
    def send(self, tensor):
        return tensor

    def recv(self, tensor):
        return tensor

    def clear_rdt_store(self):
        rdt_store = ray._private.worker.global_worker.rdt_manager.rdt_store

        with rdt_store._lock:
            assert len(rdt_store._rdt_store) > 0
            rdt_store._rdt_store.clear()

    @ray.method(concurrency_group="_ray_system")
    def block_background_thread(self):
        time.sleep(100)

    def block_main_thread(self):
        time.sleep(100)


@pytest.mark.parametrize("data_size_bytes", [100])
def test_gc_rdt_object(ray_start_regular, data_size_bytes):
    """
    For small data, GPU objects are inlined, but the actual data lives
    on the remote actor. Therefore, if we decrement the reference count
    upon inlining, we may cause the tensors on the sender actor to be
    freed before transferring to the receiver actor.

    # TODO(kevin85421): Add a test for large CPU data that is not inlined
    # after https://github.com/ray-project/ray/issues/54281 is fixed.

    对于小型数据，GPU 对象会被内联，但实际数据存放在远程 actor 上。
    因此，如果我们在内联时递减引用计数，可能会导致发送方 actor 上的
    Tensor 在传输到接收方 actor 之前就被释放。

    # TODO(kevin85421): 在 https://github.com/ray-project/ray/issues/54281
    # 修复后，添加针对未内联的大型 CPU 数据的测试。
    """
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    small_tensor = torch.randn((1,))
    cpu_data = b"1" * data_size_bytes
    data = [small_tensor, cpu_data]
    sender = actors[0]
    receiver = actors[1]

    ref1 = sender.echo.remote(data)
    ref2 = receiver.double.remote(ref1)
    ref3 = receiver.double.remote(ref1)

    result = ray.get(ref2)
    assert result[0] == pytest.approx(small_tensor * 2)
    assert result[1] == cpu_data * 2
    result = ray.get(ref3)
    assert result[0] == pytest.approx(small_tensor * 2)
    assert result[1] == cpu_data * 2

    wait_for_condition(
        lambda: ray.get(receiver.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )

    del ref1

    wait_for_condition(
        lambda: ray.get(sender.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


def test_gc_rdt_metadata(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="gloo")

    tensor = torch.randn((100, 100))
    ref = actors[0].echo.remote(tensor)
    rdt_ref_id = ref.hex()
    rdt_manager = ray._private.worker.global_worker.rdt_manager
    assert rdt_manager.is_managed_object(rdt_ref_id)
    ray.get(actors[1].double.remote(ref))
    del ref

    wait_for_condition(
        lambda: not rdt_manager.is_managed_object(rdt_ref_id),
    )


@pytest.mark.parametrize("data_size_bytes", [100])
def test_gc_del_ref_before_recv_finish(ray_start_regular, data_size_bytes):
    """
    This test deletes the ObjectRef of the GPU object before calling
    `ray.get` to ensure the receiver finishes receiving the GPU object.

    此测试在调用 `ray.get` 之前删除 GPU 对象的 ObjectRef，
    以确保接收方完成接收 GPU 对象。
    """
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    small_tensor = torch.randn((1,))
    cpu_data = b"1" * data_size_bytes
    data = [small_tensor, cpu_data]
    sender = actors[0]
    receiver = actors[1]

    ref1 = sender.echo.remote(data)
    ref2 = receiver.double.remote(ref1)

    del ref1

    result = ray.get(ref2)
    assert result[0] == pytest.approx(small_tensor * 2)
    assert result[1] == cpu_data * 2

    wait_for_condition(
        lambda: ray.get(receiver.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )
    wait_for_condition(
        lambda: ray.get(sender.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


def test_gc_intra_actor_rdt_object(ray_start_regular):
    """
    This test checks that passes a GPU object ref to the same actor multiple times.

    此测试检查将 GPU 对象引用多次传递给同一个 actor 的情况。
    """
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    small_tensor = torch.randn((1,))

    ref = actor.echo.remote(small_tensor)
    result = actor.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    result = actor.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    del ref

    wait_for_condition(
        lambda: ray.get(actor.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


def test_gc_pass_ref_to_same_and_different_actors(ray_start_regular):
    """
    This test checks that passes a GPU object ref to the same actor and a different actor.

    此测试检查将 GPU 对象引用传递给同一个 actor 和另一个不同 actor 的情况。
    """
    actor1 = GPUTestActor.remote()
    actor2 = GPUTestActor.remote()
    create_collective_group([actor1, actor2], backend="gloo")

    small_tensor = torch.randn((1,))

    ref = actor1.echo.remote(small_tensor)
    result1 = actor1.double.remote(ref)
    result2 = actor2.double.remote(ref)
    assert ray.get(result1) == pytest.approx(small_tensor * 2)
    assert ray.get(result2) == pytest.approx(small_tensor * 2)

    wait_for_condition(
        lambda: ray.get(actor2.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )

    del ref

    wait_for_condition(
        lambda: ray.get(actor1.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


def test_p2p(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    small_tensor = torch.randn((1,))
    sender = actors[0]
    receiver = actors[1]

    ref = sender.echo.remote(small_tensor)
    result = receiver.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    medium_tensor = torch.randn((500, 500))
    ref = sender.echo.remote(medium_tensor)
    result = receiver.double.remote(ref)
    assert ray.get(result) == pytest.approx(medium_tensor * 2)


def test_p2p_errors_before_group_creation(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]

    small_tensor = torch.randn((1,))
    sender = actors[0]

    with pytest.raises(
        ValueError,
        match="Actor.* does not have tensor transport GLOO available.*",
    ):
        sender.echo.remote(small_tensor)


@pytest.mark.parametrize("has_tensor_transport_method", [True, False])
def test_p2p_blocking(ray_start_regular, has_tensor_transport_method):
    """Test that p2p transfers still work when sender is blocked in another
    task. This should work whether the actor has (a) a tensor transport method
    (a method decorated with @ray.method(tensor_transport=...)) or (b) an actor-level decorator
    @ray.remote(enable_tensor_transport=True).

    测试当发送方在另一个任务中被阻塞时，p2p 传输是否仍然正常工作。
    无论 actor 具有 (a) tensor transport 方法（被 @ray.method(tensor_transport=...) 装饰的方法）
    还是 (b) actor 级装饰器 @ray.remote(enable_tensor_transport=True)，都应该正常工作。
    """

    class _GPUTestActor:
        def double(self, data):
            if isinstance(data, list):
                return [self.double(d) for d in data]
            if support_tensordict and isinstance(data, TensorDict):
                return data.apply(lambda x: x * 2)
            return data * 2

        def infinite_sleep(self, signal):
            signal.send.remote()
            while True:
                time.sleep(0.1)

    if has_tensor_transport_method:
        # Test tensor transport annotation via ray.method.
        # 测试通过 ray.method 注解 tensor transport。
        @ray.remote
        class GPUTestActor(_GPUTestActor):
            @ray.method(tensor_transport="gloo")
            def echo(self, data):
                return data

    else:
        # Test tensor transport annotation via ray.remote.
        # 测试通过 ray.remote 注解 tensor transport。
        @ray.remote(enable_tensor_transport=True)
        class GPUTestActor(_GPUTestActor):
            def echo(self, data):
                return data

    sender, receiver = GPUTestActor.remote(), GPUTestActor.remote()
    signal = SignalActor.remote()
    create_collective_group([sender, receiver], backend="gloo")
    tensor = torch.randn((500, 500))
    # If the actor does not have a tensor transport method declared, declare it
    # dynamically using .options().
    # 如果 actor 未声明 tensor transport 方法，则使用 .options() 动态声明。
    sender_fn = (
        sender.echo
        if has_tensor_transport_method
        else sender.echo.options(tensor_transport="gloo")
    )
    ref = sender_fn.remote(tensor)

    # Start a blocking task on the sender actor.
    # 在发送方 actor 上启动一个阻塞任务。
    sender.infinite_sleep.remote(signal)
    ray.get(signal.wait.remote(), timeout=10)

    # Ensure that others can still receive the object.
    # 确保其他 actor 仍然可以接收该对象。
    result = receiver.double.remote(ref)
    result = ray.get(result, timeout=10)
    assert result == pytest.approx(tensor * 2)


def test_p2p_with_cpu_data(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    sender = actors[0]
    receiver = actors[1]

    cpu_data = 123
    ref = sender.echo.remote(cpu_data)
    result = receiver.double.remote(ref)
    assert ray.get(result) == cpu_data * 2


def test_send_same_ref_to_same_actor_task_multiple_times(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    small_tensor = torch.randn((1,))
    sender = actors[0]
    receiver = actors[1]

    ref = sender.echo.remote(small_tensor)
    result = receiver.add.remote(ref, ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    wait_for_condition(
        lambda: ray.get(receiver.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


def test_send_same_ref_to_same_actor_multiple_times(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    small_tensor = torch.randn((1,))
    sender = actors[0]
    receiver = actors[1]

    ref = sender.echo.remote(small_tensor)
    result = receiver.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    result = receiver.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)


def test_intra_rdt_tensor_transfer(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    small_tensor = torch.randn((1,))

    # Intra-actor communication for pure GPU tensors
    # Actor 内部通信：纯 GPU Tensor
    ref = actor.echo.remote(small_tensor)
    result = actor.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)

    # Intra-actor communication for mixed CPU and GPU data
    # Actor 内部通信：混合 CPU 和 GPU 数据
    cpu_data = random.randint(0, 100)
    data = [small_tensor, cpu_data]
    ref = actor.echo.remote(data)
    result = actor.double.remote(ref)
    assert ray.get(result) == pytest.approx([small_tensor * 2, cpu_data * 2])

    # Intra-actor communication for multiple GPU tensors
    # Actor 内部通信：多个 GPU Tensor
    tensor1 = torch.randn((1,))
    tensor2 = torch.randn((2,))
    data = [tensor1, tensor2, cpu_data]
    ref = actor.echo.remote(data)
    result = actor.double.remote(ref)
    result = ray.get(result)

    assert result[0] == pytest.approx(tensor1 * 2)
    assert result[1] == pytest.approx(tensor2 * 2)
    assert result[2] == cpu_data * 2


def test_send_same_ref_multiple_times_intra_actor(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    small_tensor = torch.randn((1,))

    ref = actor.echo.remote(small_tensor)
    result = actor.add.remote(ref, ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)


def test_mix_cpu_gpu_data(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    tensor = torch.randn((1,))
    cpu_data = random.randint(0, 100)

    data = [tensor, cpu_data]

    sender, receiver = actors[0], actors[1]
    ref = sender.echo.remote(data)
    ref = receiver.double.remote(ref)
    result = ray.get(ref)

    assert result[0] == pytest.approx(tensor * 2)
    assert result[1] == cpu_data * 2


def test_object_in_plasma(ray_start_regular):
    """
    This test uses a CPU object that is large enough to be stored
    in plasma instead of being inlined in the gRPC message.

    此测试使用一个足够大的 CPU 对象，使其存储在 plasma 中，
    而不是内联在 gRPC 消息中。
    """
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    tensor = torch.randn((1,))
    cpu_data = b"1" * 1000 * 1000
    data = [tensor, cpu_data]

    sender, receiver = actors[0], actors[1]
    ref = sender.echo.remote(data)
    ref = receiver.double.remote(ref)
    result = ray.get(ref)

    assert result[0] == pytest.approx(tensor * 2)
    assert result[1] == cpu_data * 2


def test_multiple_tensors(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    tensor1 = torch.randn((1,))
    tensor2 = torch.randn((2,))
    if support_tensordict:
        td1 = TensorDict(
            {"action1": torch.randn((2,)), "reward1": torch.randn((2,))}, batch_size=[2]
        )
        td2 = TensorDict(
            {"action2": torch.randn((2,)), "reward2": torch.randn((2,))}, batch_size=[2]
        )
    else:
        td1 = 0
        td2 = 0
    cpu_data = random.randint(0, 100)
    data = [tensor1, tensor2, cpu_data, td1, td2]

    sender, receiver = actors[0], actors[1]
    ref = sender.echo.remote(data)
    ref = receiver.double.remote(ref)
    result = ray.get(ref)

    assert result[0] == pytest.approx(tensor1 * 2)
    assert result[1] == pytest.approx(tensor2 * 2)
    assert result[2] == cpu_data * 2
    if support_tensordict:
        assert result[3]["action1"] == pytest.approx(td1["action1"] * 2)
        assert result[3]["reward1"] == pytest.approx(td1["reward1"] * 2)
        assert result[4]["action2"] == pytest.approx(td2["action2"] * 2)
        assert result[4]["reward2"] == pytest.approx(td2["reward2"] * 2)


def test_trigger_out_of_band_tensor_transfer(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    src_actor, dst_actor = actors[0], actors[1]

    tensor = torch.tensor([1, 2, 3])
    rdt_ref = src_actor.echo.remote(tensor)
    rdt_ref_id = rdt_ref.hex()

    # Check src_actor has the GPU object
    # 检查 src_actor 是否拥有 GPU 对象
    ret_val_src = ray.get(src_actor.get_out_of_band_tensors.remote(rdt_ref_id))
    assert ret_val_src is not None
    assert len(ret_val_src) == 1
    assert torch.equal(ret_val_src[0], tensor)

    rdt_manager = ray._private.worker.global_worker.rdt_manager
    rdt_manager.add_rdt_ref(rdt_ref, src_actor, "GLOO")

    # Trigger out-of-band tensor transfer from src_actor to dst_actor.
    # 触发从 src_actor 到 dst_actor 的带外 Tensor 传输。
    task_args = (rdt_ref,)
    rdt_manager.queue_or_trigger_out_of_band_tensor_transfer(dst_actor, task_args)

    rdt_manager.set_tensor_transport_metadata_and_trigger_queued_operations(
        rdt_ref_id,
        CollectiveTransportMetadata(
            tensor_meta=[(tensor.shape, tensor.dtype)],
            tensor_device=tensor.device.type,
        ),
    )

    # Check dst_actor has the GPU object
    # 检查 dst_actor 是否拥有 GPU 对象
    ret_val_dst = ray.get(
        dst_actor.get_out_of_band_tensors.remote(rdt_ref_id, timeout=10)
    )
    assert ret_val_dst is not None
    assert len(ret_val_dst) == 1
    assert torch.equal(ret_val_dst[0], tensor)


def test_fetch_rdt_object_to_driver(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])

    # Case 1: Single tensor
    # 场景 1：单个 Tensor
    ref = actor.echo.remote(tensor1)
    assert torch.equal(ray.get(ref, _use_object_store=True), tensor1)

    # Case 2: Multiple tensors
    # 场景 2：多个 Tensor
    ref = actor.echo.remote([tensor1, tensor2])
    result = ray.get(ref, _use_object_store=True)
    assert torch.equal(result[0], tensor1)
    assert torch.equal(result[1], tensor2)

    # Case 3: Mixed CPU and GPU data
    # 场景 3：混合 CPU 和 GPU 数据
    data = [tensor1, tensor2, 7]
    ref = actor.echo.remote(data)
    result = ray.get(ref, _use_object_store=True)
    assert torch.equal(result[0], tensor1)
    assert torch.equal(result[1], tensor2)
    assert result[2] == 7


def test_invalid_tensor_transport(ray_start_regular):
    with pytest.raises(ValueError, match="Invalid tensor transport"):

        @ray.remote
        class InvalidActor:
            @ray.method(tensor_transport="invalid")
            def echo(self, data):
                return data

    actor = GPUTestActor.remote()
    with pytest.raises(ValueError, match="Invalid tensor transport"):
        actor.double.options(tensor_transport="invalid").remote(torch.randn((1,)))

    with pytest.raises(ValueError, match="Invalid tensor transport"):
        ray.put(torch.randn((1,)), _tensor_transport="invalid")


@pytest.mark.skipif(
    not support_tensordict,
    reason="tensordict is not supported on this platform",  # tensordict 在此平台上不受支持
)
def test_tensordict_transfer(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    td = TensorDict(
        {"action": torch.randn((2,)), "reward": torch.randn((2,))}, batch_size=[2]
    )
    sender, receiver = actors[0], actors[1]
    ref = sender.echo.remote(td)
    result = receiver.double.remote(ref)
    td_result = ray.get(result)

    assert td_result["action"] == pytest.approx(td["action"] * 2)
    assert td_result["reward"] == pytest.approx(td["reward"] * 2)


@pytest.mark.skipif(
    not support_tensordict,
    reason="tensordict is not supported on this platform",  # tensordict 在此平台上不受支持
)
def test_nested_tensordict(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    inner_td = TensorDict(
        {"action": torch.randn((2,)), "reward": torch.randn((2,))}, batch_size=[2]
    )
    outer_td = TensorDict(
        {"inner_td": inner_td, "test": torch.randn((2,))}, batch_size=[2]
    )
    sender = actors[0]
    receiver = actors[1]
    rdt_ref = sender.echo.remote(outer_td)
    ret_val_src = ray.get(receiver.double.remote(rdt_ref))
    assert ret_val_src is not None
    assert torch.equal(ret_val_src["inner_td"]["action"], inner_td["action"] * 2)
    assert torch.equal(ret_val_src["inner_td"]["reward"], inner_td["reward"] * 2)
    assert torch.equal(ret_val_src["test"], outer_td["test"] * 2)


@pytest.mark.skipif(
    not support_tensordict,
    reason="tensordict is not supported on this platform",  # tensordict 在此平台上不受支持
)
def test_tensor_extracted_from_tensordict_in_rdt_store(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    td = TensorDict(
        {"action": torch.randn((2,)), "reward": torch.randn((2,))}, batch_size=[2]
    ).to("cpu")
    rdt_ref = actor.echo.remote(td)

    # Since the tensor is extracted from the tensordict, the `ret_val_src` will be a list of tensors
    # instead of a tensordict.
    # 由于 Tensor 是从 tensordict 中提取的，`ret_val_src` 将是一个 Tensor 列表，
    # 而不是一个 tensordict。
    ret_val_src = ray.get(actor.get_out_of_band_tensors.remote(rdt_ref.hex()))
    assert ret_val_src is not None
    assert len(ret_val_src) == 2
    assert torch.equal(ret_val_src[0], td["action"])
    assert torch.equal(ret_val_src[1], td["reward"])


@pytest.mark.parametrize("enable_tensor_transport", [True, False])
def test_dynamic_tensor_transport_via_options(
    ray_start_regular, enable_tensor_transport
):
    """Test that tensor_transport can be set dynamically via .options() at call
    time, if enable_tensor_transport is set to True in @ray.remote.

    测试如果 @ray.remote 中 enable_tensor_transport 设置为 True，
    则可以通过 .options() 在调用时动态设置 tensor_transport。
    """

    class TestActor:
        def __init__(self):
            pass

        def normal_method(self):
            return "normal"

        def tensor_method(self):
            return torch.randn(5, 5)

        def double(self, data):
            return data * 2

    if enable_tensor_transport:
        TestActor = ray.remote(enable_tensor_transport=True)(TestActor)
    else:
        TestActor = ray.remote(TestActor)

    # Create actor without any tensor_transport decorators
    # 创建不带任何 tensor_transport 装饰器的 actor
    sender = TestActor.remote()
    receiver = TestActor.remote()
    create_collective_group([sender, receiver], backend="gloo")

    # Test normal method call
    # 测试普通方法调用
    result = ray.get(sender.normal_method.remote())
    assert result == "normal"

    # Test method call with tensor_transport specified via .options()
    # 测试通过 .options() 指定 tensor_transport 的方法调用
    if enable_tensor_transport:
        # If enable_tensor_transport is set to True, then it's okay to use
        # dynamic tensor_transport.
        # 如果 enable_tensor_transport 设置为 True，则可以使用动态 tensor_transport。
        ref = sender.tensor_method.options(tensor_transport="gloo").remote()
        tensor = ray.get(ref, _use_object_store=True)
        result = ray.get(receiver.double.remote(ref), _use_object_store=True)
        assert result == pytest.approx(tensor * 2)
    else:
        # If enable_tensor_transport is not set, then user cannot use
        # dynamic tensor_transport.
        # 如果未设置 enable_tensor_transport，则用户不能使用动态 tensor_transport。
        with pytest.raises(
            ValueError,
            match='Currently, methods with .options\\(tensor_transport="GLOO"\\) are not supported when enable_tensor_transport=False. Please set @ray.remote\\(enable_tensor_transport=True\\) on the actor class definition.',
        ):
            ref = sender.tensor_method.options(tensor_transport="gloo").remote()


def test_app_error_inter_actor(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    src_actor, dst_actor = actors[0], actors[1]

    # Make sure the receiver can receive an exception from the sender.
    # 确保接收方能够接收来自发送方的异常。
    ref = src_actor.fail.options(tensor_transport="gloo").remote("test_app_error")
    with pytest.raises(Exception, match="test_app_error"):
        ray.get(dst_actor.double.remote(ref))

    # Make sure the sender and receiver do not hang.
    # 确保发送方和接收方不会挂起。
    small_tensor = torch.randn((1,))
    ref = src_actor.echo.remote(small_tensor)
    result = dst_actor.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)


def test_app_error_intra_actor(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    # Make sure the receiver can receive an exception from the sender.
    # 确保接收方能够接收来自发送方的异常。
    ref = actor.fail.options(tensor_transport="gloo").remote("test_app_error")
    with pytest.raises(Exception, match="test_app_error"):
        ray.get(actor.double.remote(ref))

    # Make sure the sender and receiver do not hang.
    # 确保发送方和接收方不会挂起。
    small_tensor = torch.randn((1,))
    ref = actor.echo.remote(small_tensor)
    result = actor.double.remote(ref)
    assert ray.get(result) == pytest.approx(small_tensor * 2)


def test_app_error_fetch_to_driver(ray_start_regular):
    actor = GPUTestActor.remote()
    create_collective_group([actor], backend="gloo")

    ref = actor.fail.options(tensor_transport="gloo").remote("test_app_error")
    with pytest.raises(Exception, match="test_app_error"):
        ray.get(ref, _use_object_store=True)

    # Make sure the driver can receive an exception from the actor.
    # 确保 driver 能够接收来自 actor 的异常。
    small_tensor = torch.tensor([1, 2, 3])
    ref = actor.echo.remote(small_tensor)
    assert torch.equal(ray.get(ref, _use_object_store=True), small_tensor)


@ray.remote
class FailingRDTActor:
    def __init__(self):
        self.attempts = 0

    @ray.method(
        tensor_transport="gloo", max_task_retries=1, retry_exceptions=[ValueError]
    )
    def fail_first_attempt(self):
        self.attempts += 1
        if self.attempts == 1:
            raise ValueError("first-attempt failure")
        return torch.tensor([1, 2, 3])

    @ray.method(
        tensor_transport="gloo", max_task_retries=1, retry_exceptions=[ValueError]
    )
    def rdt_obj_always_fails(self):
        self.attempts += 1
        raise ValueError("permanent failure")

    def consume(self, tensor):
        return tensor

    def get_num_rdt_objects(self):
        return ray._private.worker.global_worker.rdt_manager.rdt_store.get_num_objects()


def test_rdt_retry_then_succeeds(ray_start_regular):
    """
    Retryable exception on first attempt, success on second
    Only one entry should be in the RDTStore.

    第一次尝试时出现可重试异常，第二次成功。
    RDTStore 中应只有一个条目。
    """
    sender = FailingRDTActor.remote()
    receiver = FailingRDTActor.remote()
    create_collective_group([sender, receiver], backend="gloo")

    ref = sender.fail_first_attempt.remote()
    result = ray.get(receiver.consume.remote(ref))
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # Sender should hold one primary entry for this ref
    # 发送方应为此引用持有一个主条目
    assert ray.get(sender.get_num_rdt_objects.remote()) == 1


def test_rdt_retry_fetch_through_obj_store(ray_start_regular):
    """
    Retryable exception on first attempt, successful fetch to driver on second

    第一次尝试时出现可重试异常，第二次成功获取到 driver。
    """
    sender = FailingRDTActor.remote()
    create_collective_group([sender], backend="gloo")

    ref = sender.fail_first_attempt.remote()
    assert torch.equal(ray.get(ref, _use_object_store=True), torch.tensor([1, 2, 3]))


def test_rdt_retries_exhausted_raises(ray_start_regular):
    """
    When all retries fail, the user's exception must propagate to the
    consumer via the CPU path (no direct_transport_metadata is set on the
    final reply, so the consumer sees the error when deserializing the arg).

    当所有重试都失败时，用户的异常必须通过 CPU 路径传播到消费方
    （最终回复上未设置 direct_transport_metadata，因此消费方在反序列化参数时看到错误）。
    """
    sender = FailingRDTActor.remote()
    receiver = FailingRDTActor.remote()
    create_collective_group([sender, receiver], backend="gloo")

    ref = sender.rdt_obj_always_fails.remote()
    with pytest.raises(Exception, match="permanent failure"):
        ray.get(receiver.consume.remote(ref))


def test_write_after_save(ray_start_regular):
    """Check that an actor can safely write to a tensor after saving it to its
    local state by calling `ray.experimental.wait_tensor_freed`.

    检查 actor 在通过调用 `ray.experimental.wait_tensor_freed` 将 Tensor
    保存到其本地状态后，是否可以安全地写入该 Tensor。
    """

    @ray.remote(enable_tensor_transport=True)
    class GPUTestActor:
        @ray.method(tensor_transport="gloo")
        def save(self, data: torch.Tensor):
            # Save the tensor to the actor's local state.
            # 将 Tensor 保存到 actor 的本地状态。
            self.data = data
            return data

        def receive(self, data: torch.Tensor):
            return data

        def increment_saved(self):
            ray.experimental.wait_tensor_freed(self.data)
            # Write to the saved tensor.
            # 写入已保存的 Tensor。
            self.data += 1
            return self.data

    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    medium_tensor = torch.randn((500, 500))
    sender, receiver = actors
    ref = sender.save.remote(medium_tensor)
    # Sender writes to the GPU object while Ray sends the object to a receiver
    # task in the background.
    # 发送方在 Ray 后台将对象发送给接收方任务时，写入 GPU 对象。
    tensor1 = sender.increment_saved.remote()
    tensor2 = receiver.receive.remote(ref)

    # The sender task should not have returned yet because the ObjectRef is
    # still in scope.
    # 发送方任务尚未返回，因为 ObjectRef 仍在作用域内。
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(tensor1, timeout=1)

    del ref
    # Check that Ray completed the transfer of the original tensor before the
    # sender writes to it.
    # 检查 Ray 在发送方写入之前是否已完成原始 Tensor 的传输。
    assert torch.allclose(ray.get(tensor1), medium_tensor + 1)
    assert torch.allclose(ray.get(tensor2), medium_tensor)


def test_wait_tensor_freed(ray_start_regular):
    """Unit test for ray.experimental.wait_tensor_freed. Check that the call
    returns when the tensor has been freed from the GPU object store.

    ray.experimental.wait_tensor_freed 的单元测试。检查当 Tensor 从 GPU
    对象存储中被释放时，该调用是否返回。
    """
    rdt_store = ray.worker.global_worker.rdt_manager.rdt_store
    obj_id = "random_id"
    tensor = torch.randn((1,))
    rdt_store.add_object_primary(obj_id, [tensor], "GLOO")

    assert rdt_store.has_object(obj_id)
    with pytest.raises(TimeoutError):
        ray.experimental.wait_tensor_freed(tensor, timeout=1)
    assert rdt_store.has_object(obj_id)

    # Simulate garbage collection in a background thread.
    # 在后台线程中模拟垃圾回收。
    def gc():
        time.sleep(0.1)
        rdt_store.pop_object(obj_id)

    gc_thread = threading.Thread(target=gc)
    gc_thread.start()
    # Now the wait_tensor_freed call should be able to return.
    # 现在 wait_tensor_freed 调用应该能够返回。
    ray.experimental.wait_tensor_freed(tensor)
    gc_thread.join()
    assert not rdt_store.has_object(obj_id)


def test_wait_tensor_freed_double_tensor(ray_start_regular):
    """Unit test for ray.experimental.wait_tensor_freed when multiple objects
    contain the same tensor.

    ray.experimental.wait_tensor_freed 的单元测试，当多个对象包含同一 Tensor 时。
    """
    rdt_store = ray.worker.global_worker.rdt_manager.rdt_store
    obj_id1 = "random_id1"
    obj_id2 = "random_id2"
    tensor = torch.randn((1,))
    rdt_store.add_object_primary(obj_id1, [tensor], "GLOO")
    rdt_store.add_object_primary(obj_id2, [tensor], "GLOO")

    assert rdt_store.has_object(obj_id1)
    assert rdt_store.has_object(obj_id2)
    with pytest.raises(TimeoutError):
        ray.experimental.wait_tensor_freed(tensor, timeout=1)
    assert rdt_store.has_object(obj_id1)
    assert rdt_store.has_object(obj_id2)

    # Simulate garbage collection in a background thread.
    # 在后台线程中模拟垃圾回收。
    def gc(obj_id):
        time.sleep(0.1)
        rdt_store.pop_object(obj_id)

    # Free one object. Tensor should still be stored.
    # 释放一个对象。Tensor 应仍然被存储。
    gc_thread = threading.Thread(target=gc, args=(obj_id1,))
    gc_thread.start()
    with pytest.raises(TimeoutError):
        ray.experimental.wait_tensor_freed(tensor, timeout=1)
    gc_thread.join()
    assert not rdt_store.has_object(obj_id1)

    # Free the other object. Now the wait_tensor_freed call should be able to
    # return.
    # 释放另一个对象。现在 wait_tensor_freed 调用应该能够返回。
    gc_thread = threading.Thread(target=gc, args=(obj_id2,))
    gc_thread.start()
    ray.experimental.wait_tensor_freed(tensor)
    gc_thread.join()
    assert not rdt_store.has_object(obj_id2)


def test_send_back_and_dst_warning(ray_start_regular):
    # Test warning when object is sent back to the src actor and to dst actors
    # 测试当对象被发送回源 actor 和目标 actor 时的警告
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")

    src_actor, dst_actor = actors[0], actors[1]

    tensor = torch.tensor([1, 2, 3])

    warning_message = r"RDT ObjectRef\(.+\)"

    with pytest.warns(UserWarning, match=warning_message):
        t = src_actor.echo.remote(tensor)
        t1 = src_actor.echo.remote(t)  # Sent back to the source actor
        t1 = src_actor.echo.remote(t)  # 发送回源 actor
        t2 = dst_actor.echo.remote(t)  # Also sent to another actor
        t2 = dst_actor.echo.remote(t)  # 同时发送到另一个 actor
        ray.get([t1, t2], _use_object_store=True)

    # Second transmission of ObjectRef `t` to `dst_actor` should not trigger a warning
    # Verify no `pytest.warns` context is used here because no warning should be raised
    # ObjectRef `t` 第二次传输到 `dst_actor` 不应触发警告
    # 验证此处未使用 `pytest.warns` 上下文，因为不应触发任何警告
    t3 = dst_actor.echo.remote(t)
    ray.get(t3, _use_object_store=True)


def test_duplicate_objectref_transfer(ray_start_regular):
    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="gloo")
    actor0, actor1 = actors[0], actors[1]

    small_tensor = torch.randn((1,))

    # Store the original value for comparison
    # 保存原始值用于比较
    original_value = small_tensor

    ref = actor0.echo.remote(small_tensor)

    # Pass the same ref to actor1 twice
    # 将同一引用传递给 actor1 两次
    result1 = actor1.increment.remote(ref)
    result2 = actor1.increment.remote(ref)

    # Both should return original_value + 1 because each increment task should receive the same object value.
    # 两者都应返回 original_value + 1，因为每个 increment 任务应接收到相同的对象值。
    val1 = ray.get(result1)
    val2 = ray.get(result2)

    # Check for correctness
    # 检查正确性
    assert val1 == pytest.approx(
        original_value + 1
    ), f"Result1 incorrect: got {val1}, expected {original_value + 1}"
    assert val2 == pytest.approx(
        original_value + 1
    ), f"Result2 incorrect: got {val2}, expected {original_value + 1}"

    # Additional check: results should be equal (both got clean copies)
    # 附加检查：结果应该相等（两者都获得了干净的副本）
    assert val1 == pytest.approx(
        val2
    ), f"Results differ: result1={val1}, result2={val2}"


def test_transfer_from_not_actor_creator(ray_start_regular):
    @ray.remote
    class Actor:
        @ray.method(tensor_transport="gloo")
        def create(self):
            return torch.tensor([1, 2, 3])

        def consume(self, obj):
            return obj

        def do_transfer(self, a1, a2):
            create_collective_group([a1, a2], backend="torch_gloo")
            return ray.get(a1.consume.remote(a2.create.remote()))

    actor = [Actor.remote() for _ in range(3)]
    assert ray.get(actor[2].do_transfer.remote(actor[0], actor[1])) == pytest.approx(
        torch.tensor([1, 2, 3])
    )


def test_send_fails(ray_start_regular):
    actors = [ErrorActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="torch_gloo")

    # The gpu object will be gone when we trigger the transfer
    # so the send will error out
    # 当我们触发传输时，GPU 对象将不存在，
    # 因此发送将报错
    rdt_ref = actors[0].send.remote(torch.randn((100, 100)))
    ray.get(actors[0].clear_rdt_store.remote())
    result_ref = actors[1].recv.remote(rdt_ref)

    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(result_ref)


def test_send_actor_dies_before_creating(ray_start_regular):
    actors = [ErrorActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="torch_gloo")

    # Block the main thread so the object doesn't get created before the kill
    # 阻塞主线程，使对象在被杀死之前不会被创建
    actors[0].block_main_thread.remote()
    gpu_obj_ref = actors[0].send.remote(torch.randn(100, 100))
    result_ref = actors[1].recv.remote(gpu_obj_ref)
    ray.kill(actors[0])

    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(result_ref)


def test_send_actor_dies_before_sending(ray_start_regular):
    actors = [ErrorActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="torch_gloo")

    rdt_ref = actors[0].send.remote(torch.randn(100, 100))
    # Wait for the object to actually be created on the sender
    # 等待对象在发送方上实际创建
    ray.wait([rdt_ref])
    # Block the background thread before triggering the transfer
    # so the send doesn't happen before the actor is killed
    # 在触发传输之前阻塞后台线程，
    # 使发送不会在 actor 被杀死之前发生
    actors[0].block_background_thread.remote()
    result_ref = actors[1].recv.remote(rdt_ref)
    ray.kill(actors[0])

    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(result_ref)


def test_recv_actor_dies(ray_start_regular, caplog, propagate_logs):
    actors = [ErrorActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="torch_gloo")

    # Do a transfer with the receiver's background thread blocked,
    # so the recv doesn't happen before the actor is killed
    # 在接收方后台线程被阻塞的情况下进行传输，
    # 使接收不会在 actor 被杀死之前发生
    rdt_ref = actors[0].send.remote(torch.randn((100, 100)))
    actors[1].block_background_thread.remote()
    result_ref = actors[1].recv.remote(rdt_ref)
    ray.kill(actors[1])

    def check_logs():
        records = caplog.records
        return any(
            record.levelno == logging.ERROR
            and re.search(r"RDT transfer with.*failed", record.message)
            for record in records
        ) and any(
            record.levelno == logging.ERROR
            and "Destroyed collective group" in record.message
            for record in records
        )

    wait_for_condition(check_logs)

    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(result_ref)
    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(actors[0].recv.remote(1))


@pytest.mark.skip(
    "Lineage Reconstruction currently results in a check failure with RDT"  # Lineage Reconstruction 目前在 RDT 中会导致检查失败
)
def test_rdt_lineage_reconstruction(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1)
    worker_to_kill = cluster.add_node(num_cpus=1, resources={"to_restart": 1})

    @ray.remote(max_restarts=1, max_task_retries=1, resources={"to_restart": 1})
    class RecvRestartableActor:
        def recv(self, obj):
            return obj

    send_actor = GPUTestActor.remote()
    recv_actor = RecvRestartableActor.remote()
    create_collective_group([send_actor, recv_actor], backend="gloo")

    one_mb_tensor = torch.randn((1024 * 1024,))
    ref = recv_actor.recv.remote(send_actor.echo.remote(one_mb_tensor))
    ray.wait([ref], fetch_local=False)
    cluster.remove_node(worker_to_kill, allow_graceful=False)
    cluster.add_node(num_cpus=1, resources={"to_restart": 1})
    assert ray.get(ref).nbytes >= (1024 * 1024)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
