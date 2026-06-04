import sys

import pytest
import torch

import ray
from ray._common.test_utils import SignalActor, wait_for_condition
from ray.experimental import set_target_for_ref
from ray.experimental.rdt.util import get_tensor_transport_manager


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class GPUTestActor:
    def __init__(self):
        self.reserved_tensor1 = torch.tensor([1, 2, 3]).to("cuda")
        self.reserved_tensor2 = torch.tensor([4, 5, 6]).to("cuda")
        self.reserved_tensor3 = torch.tensor([7, 8, 9]).to("cuda")

    @ray.method(tensor_transport="nixl")
    def echo(self, data, device):
        return data.to(device)

    def sum(self, data, device):
        assert data.device.type == device
        return data.sum().item()

    def produce(self, tensors):
        refs = []
        for t in tensors:
            refs.append(ray.put(t, _tensor_transport="nixl"))
        return refs

    def consume_with_nixl(self, refs):
        tensors = [ray.get(ref) for ref in refs]
        sum = 0
        for t in tensors:
            assert t.device.type == "cuda"
            sum += t.sum().item()
        return sum

    def consume_with_object_store(self, refs):
        tensors = [ray.get(ref, _use_object_store=True) for ref in refs]
        sum = 0
        for t in tensors:
            assert t.device.type == "cuda"
            sum += t.sum().item()
        return sum

    def gc(self):

        tensor = torch.tensor([1, 2, 3]).to("cuda")
        ref = ray.put(tensor, _tensor_transport="nixl")
        obj_id = ref.hex()
        rdt_manager = ray._private.worker.global_worker.rdt_manager
        nixl_transport = get_tensor_transport_manager("NIXL")

        assert rdt_manager.rdt_store.has_tensor(tensor)
        assert rdt_manager.is_managed_object(obj_id)
        assert obj_id in nixl_transport._managed_meta_nixl
        # Tensor-level metadata counting: the tensor should have metadata_count=1
        # Tensor 级别的元数据计数：该 tensor 应具有 metadata_count=1
        key = tensor.untyped_storage().data_ptr()
        assert key in nixl_transport._tensor_desc_cache
        assert nixl_transport._tensor_desc_cache[key].metadata_count == 1

        del ref

        rdt_manager.rdt_store.wait_tensor_freed(tensor, timeout=10)
        assert not rdt_manager.rdt_store.has_tensor(tensor)
        assert not rdt_manager.is_managed_object(obj_id)
        assert obj_id not in nixl_transport._managed_meta_nixl
        assert key not in nixl_transport._tensor_desc_cache
        return "Success"

    @ray.method(tensor_transport="nixl")
    def send_dict1(self):
        return {"round1-1": self.reserved_tensor1, "round1-2": self.reserved_tensor2}

    @ray.method(tensor_transport="nixl")
    def send_dict2(self):
        return {"round2-1": self.reserved_tensor1, "round2-3": self.reserved_tensor3}

    def sum_dict(self, dict):
        return sum(v.sum().item() for v in dict.values())

    def get_num_rdt_objects(self):
        rdt_manager = ray._private.worker.global_worker.rdt_manager
        return rdt_manager.rdt_store.get_num_objects()

    def get_num_managed_meta_nixl(self):

        return get_tensor_transport_manager("NIXL")._get_num_managed_meta_nixl()

    def put_shared_tensor_lists(self):
        """Create two tensor lists that share a common tensor and put them with NIXL transport.

创建两个共享同一 tensor 的 tensor 列表，并使用 NIXL 传输方式放入。"""
        t1 = torch.tensor([1, 2, 3]).to("cuda")
        t2 = torch.tensor([4, 5, 6]).to("cuda")
        t3 = torch.tensor([7, 8, 9]).to("cuda")

        list1 = [t1, t2]
        list2 = [t2, t3]

        ref1 = ray.put(list1, _tensor_transport="nixl")
        # Nixl itself doesn't handle duplicate memory registrations,
        # hence this call would fail without proper deduplication.
        # NIXL 本身不处理重复的内存注册，
        # 因此如果没有适当的去重机制，此调用将会失败。
        ref2 = ray.put(list2, _tensor_transport="nixl")

        return ref1, ref2

    @ray.method(concurrency_group="_ray_system")
    def block_background_thread(self, signal_actor):
        ray.get(signal_actor.wait.remote())

    def borrow_and_sum(self, ref_list):
        return ray.get(ref_list[0]).sum().item()

    def block_main_thread(self, signal_actor):
        ray.get(signal_actor.wait.remote())


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_ray_get_rdt_ref_created_by_actor_task(ray_start_regular):
    actor = GPUTestActor.remote()
    tensor = torch.tensor([1, 2, 3]).to("cuda")
    ref1 = actor.echo.remote(tensor, "cuda")
    ref2 = actor.echo.remote(tensor, "cuda")
    ref3 = actor.echo.remote(tensor, "cuda")

    # Test ray.get with default tensor transport, should use nixl here.
    # TODO: Verify it's using the correct tensor transport.
    # 测试使用默认 tensor 传输方式的 ray.get，此处应使用 NIXL。
    # TODO：验证其使用了正确的 tensor 传输方式。
    assert torch.equal(ray.get(ref1), tensor)

    # # Test ray.get with nixl tensor transport
    # # 测试使用 NIXL tensor 传输方式的 ray.get
    assert torch.equal(ray.get(ref2), tensor)

    # # Test ray.get with object store tensor transport
    # # 测试使用 object store tensor 传输方式的 ray.get
    assert torch.equal(ray.get(ref3, _use_object_store=True), tensor)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_p2p(ray_start_regular):
    num_actors = 2
    actors = [GPUTestActor.remote() for _ in range(num_actors)]

    src_actor, dst_actor = actors[0], actors[1]

    # Create test tensor
    # 创建测试 tensor
    tensor = torch.tensor([1, 2, 3])

    tensor1 = torch.tensor([4, 5, 6])

    # Test GPU to GPU transfer
    # 测试 GPU 到 GPU 的传输
    ref = src_actor.echo.remote(tensor, "cuda")

    # Trigger tensor transfer from src to dst actor
    # 触发从源 actor 到目标 actor 的 tensor 传输
    result = dst_actor.sum.remote(ref, "cuda")
    assert tensor.sum().item() == ray.get(result)

    # Test CPU to CPU transfer
    # 测试 CPU 到 CPU 的传输
    ref1 = src_actor.echo.remote(tensor1, "cpu")
    result1 = dst_actor.sum.remote(ref1, "cpu")
    assert tensor1.sum().item() == ray.get(result1)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_intra_rdt_tensor_transfer(ray_start_regular):
    actor = GPUTestActor.remote()

    tensor = torch.tensor([1, 2, 3])

    # Intra-actor communication for pure GPU tensors
    # 纯 GPU tensor 的 actor 内部通信
    ref = actor.echo.remote(tensor, "cuda")
    result = actor.sum.remote(ref, "cuda")
    assert tensor.sum().item() == ray.get(result)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_put_and_get_object_with_nixl(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]
    src_actor, dst_actor = actors[0], actors[1]
    tensor1 = torch.tensor([1, 2, 3]).to("cuda")
    tensor2 = torch.tensor([4, 5, 6, 0]).to("cuda")
    tensor3 = torch.tensor([7, 8, 9, 0, 0]).to("cuda")
    tensors = [tensor1, tensor2, tensor3]
    ref = src_actor.produce.remote(tensors)
    ref1 = dst_actor.consume_with_nixl.remote(ref)
    result1 = ray.get(ref1)
    assert result1 == 45


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_put_and_get_object_with_object_store(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]
    src_actor, dst_actor = actors[0], actors[1]
    tensor1 = torch.tensor([1, 2, 3]).to("cuda")
    tensor2 = torch.tensor([4, 5, 6, 0]).to("cuda")
    tensor3 = torch.tensor([7, 8, 9, 0, 0]).to("cuda")
    tensors = [tensor1, tensor2, tensor3]
    ref = src_actor.produce.remote(tensors)
    ref1 = dst_actor.consume_with_object_store.remote(ref)
    result1 = ray.get(ref1)
    assert result1 == 45


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_put_gc(ray_start_regular):
    actor = GPUTestActor.remote()
    ref = actor.gc.remote()
    assert ray.get(ref) == "Success"


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_send_duplicate_tensor(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]
    src_actor, dst_actor = actors[0], actors[1]
    ref1 = src_actor.send_dict1.remote()
    result1 = dst_actor.sum_dict.remote(ref1)
    assert ray.get(result1) == 21
    ref2 = src_actor.send_dict1.remote()
    result2 = dst_actor.sum_dict.remote(ref2)
    assert ray.get(result2) == 21

    del ref1
    del ref2
    wait_for_condition(
        lambda: ray.get(src_actor.get_num_rdt_objects.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )
    wait_for_condition(
        lambda: ray.get(src_actor.get_num_managed_meta_nixl.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_abort_sender_dies_before_creating(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]

    # Trigger transfer and kill sender before the receiver starts receiving
    # 触发传输并在接收方开始接收之前杀死发送方
    signal_actor = SignalActor.remote()
    actors[0].block_main_thread.remote(signal_actor)
    ref = actors[0].echo.remote(torch.randn((100, 100)), "cuda")
    result = actors[1].sum.remote(ref, "cuda")
    ray.kill(actors[0])

    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(result)

    # Try a transfer with actor[1] receiving again
    # 尝试再次让 actor[1] 作为接收方进行传输
    new_actor = GPUTestActor.remote()
    ref = new_actor.echo.remote(torch.tensor([4, 5, 6]), "cuda")
    result = actors[1].sum.remote(ref, "cuda")
    assert ray.get(result) == 15


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_abort_sender_dies_before_sending(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]

    """
    1. Block background thread on receiver so receive doesn't start
    2. Wait until the object is created so the transfer gets triggered
    3. Kill the sender
    4. Unblock the receiver

    1. 阻塞接收方的后台线程，使接收不会开始
    2. 等待直到对象被创建，从而触发传输
    3. 杀死发送方
    4. 解除接收方的阻塞
    """

    signal_actor = SignalActor.remote()
    actors[1].block_background_thread.remote(signal_actor)
    ref = actors[0].echo.remote(torch.randn((100, 100)), "cuda")
    result = actors[1].sum.remote(ref, "cuda")
    ray.wait([ref])
    ray.kill(actors[0])
    signal_actor.send.remote()

    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(result)

    exc_str = str(excinfo.value)
    assert "nixlBackendError" in exc_str and "The source actor may have died" in exc_str

    # Try a transfer with actor[1] receiving again
    # 尝试再次让 actor[1] 作为接收方进行传输
    new_actor = GPUTestActor.remote()
    ref = new_actor.echo.remote(torch.tensor([4, 5, 6]), "cuda")
    result = actors[1].sum.remote(ref, "cuda")
    assert ray.get(result) == 15


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_del_before_creating(ray_start_regular):
    """
    Blocking the main thread until we free the object from the reference counter.
    Then unblocking the actor's main thread so the object can be created and then
    asserting that the object was actually freed.

    阻塞主线程直到从引用计数器中释放该对象。
    然后解除对 actor 主线程的阻塞，使对象可以被创建，
    最后断言该对象确实已被释放。
    """
    signal_actor = SignalActor.remote()
    actor = GPUTestActor.remote()
    actor.block_main_thread.remote(signal_actor)
    ref = actor.echo.remote(torch.tensor([4, 5, 6]), "cuda")
    obj_id = ref.hex()
    del ref
    ray.get(signal_actor.send.remote())

    wait_for_condition(
        lambda: ray._private.worker.global_worker.rdt_manager.get_rdt_metadata(obj_id)
        is None,
    )
    wait_for_condition(
        lambda: ray.get(actor.get_num_rdt_objects.remote()) == 0,
    )


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_owner_gets_from_launched_task(ray_start_regular):
    actor = GPUTestActor.remote()
    tensor = torch.randn((100, 100))

    ref = actor.echo.remote(tensor, "cuda")
    assert torch.equal(ray.get(ref), tensor.to("cuda"))


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_out_of_order_actors(ray_start_regular):
    @ray.remote(num_cpus=0, num_gpus=1, max_concurrency=10)
    class GPUTestActor:
        def __init__(self):
            self.tensor = torch.tensor([4, 5, 6], device="cuda")

        @ray.method(tensor_transport="nixl")
        async def get_tensor(self):
            return self.tensor

        async def sum(self, data):
            return data.sum().item()

    actors = [GPUTestActor.remote() for _ in range(2)]
    results = []
    for _ in range(100):
        ref = actors[0].get_tensor.remote()
        result = actors[1].sum.remote(ref)
        results.append(result)
    results = ray.get(results)
    assert sum(results) == 1500


@pytest.mark.skip(
    "If the tensor metadata doesn't exist at the time of borrowing, this will fail."
    # 如果在借用时 tensor 元数据不存在，此测试将会失败。
)
@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_borrow_after_abort(ray_start_regular):
    actors = [GPUTestActor.remote() for _ in range(2)]
    nixl_ref = actors[0].echo.remote(torch.tensor([4, 5, 6]), "cuda")
    assert ray.get(actors[1].borrow_and_sum.remote([nixl_ref])) == 15


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_shared_tensor_deduplication(ray_start_regular):
    """
    Test that tensors shared across multiple lists are properly deduplicated.

    Creates list1 = [T1, T2] and list2 = [T2, T3] where T2 is shared.

    测试跨多个列表共享的 tensor 是否被正确去重。
    创建 list1 = [T1, T2] 和 list2 = [T2, T3]，其中 T2 是共享的。
    """
    actor = GPUTestActor.remote()
    ray.get(actor.put_shared_tensor_lists.remote())


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_agent_reuse(ray_start_regular):
    """
    We reuse nixl remote agent by default. The receiver should successfully receive
    all tensors while the sender may trigger GC in between.

    默认情况下我们复用 NIXL 远程代理。接收方应成功接收所有 tensor，
    而发送方可能会在中间触发 GC。
    """
    actors = [GPUTestActor.remote() for _ in range(2)]
    src_actor, dst_actor = actors[0], actors[1]

    ref1 = src_actor.echo.remote(torch.tensor([1, 2, 3]).to("cuda"), "cuda")
    assert ray.get(dst_actor.sum.remote(ref1, "cuda")) == 6

    # Trigger another transfer. The receiver successfully gets
    # the latest tensor (nixl agent is reused internally).
    # 触发另一次传输。接收方成功获取最新的 tensor（NIXL 代理在内部被复用）。
    ref2 = src_actor.echo.remote(torch.tensor([4, 5, 6]).to("cuda"), "cuda")
    assert ray.get(dst_actor.sum.remote(ref2, "cuda")) == 15

    del ref1, ref2

    # Wait for GC to free the tensors on the sender.
    # 等待 GC 释放发送方上的 tensor。
    wait_for_condition(
        lambda: ray.get(src_actor.get_num_managed_meta_nixl.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )

    # Transfer after GC. The receiver successfully gets
    # the latest tensor (nixl agent is reset internally).
    # GC 后的传输。接收方成功获取最新的 tensor（NIXL 代理在内部被重置）。
    ref3 = src_actor.echo.remote(torch.tensor([7, 8, 9]).to("cuda"), "cuda")
    assert ray.get(dst_actor.sum.remote(ref3, "cuda")) == 24


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_agent_reuse_with_partial_tensors(ray_start_regular):
    """
    We reuse nixl remote agent by default. The receiver should successfully choose
    and receive part of the tensors.

    默认情况下我们复用 NIXL 远程代理。接收方应成功选择并接收部分 tensor。
    """
    actors = [GPUTestActor.remote() for _ in range(2)]
    src_actor, dst_actor = actors[0], actors[1]

    ref1 = src_actor.echo.remote(torch.tensor([1, 2, 3, 4, 5, 6]).to("cuda"), "cuda")
    assert ray.get(dst_actor.sum.remote(ref1, "cuda")) == 21

    del ref1

    # Wait for GC to free the tensors on the sender.
    # 等待 GC 释放发送方上的 tensor。
    wait_for_condition(
        lambda: ray.get(src_actor.get_num_managed_meta_nixl.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )

    # Create the second tensor at the sender. The memory address of
    # this tensor may overlap with the first tensor (de-registered).
    # 在发送方创建第二个 tensor。此 tensor 的内存地址可能与第一个 tensor（已注销注册）重叠。
    ref2 = src_actor.echo.remote(torch.tensor([1, 2, 3]).to("cuda"), "cuda")

    # Create the third tensor at the sender. The memory address of
    # this tensor may overlap with the first tensor (de-registered).
    # 在发送方创建第三个 tensor。此 tensor 的内存地址可能与第一个 tensor（已注销注册）重叠。
    ref3 = src_actor.echo.remote(torch.tensor([4, 5, 6]).to("cuda"), "cuda")
    # Trigger the transfer. The receiver successfully gets
    # the third tensor (nixl agent is reset internally).
    # 触发传输。接收方成功获取第三个 tensor（NIXL 代理在内部被重置）。
    assert ray.get(dst_actor.sum.remote(ref3, "cuda")) == 15

    del ref2, ref3


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_storage_level_overlapping_views_reference_count(ray_start_regular):
    """Test that two overlapping tensors sharing the same underlying storage produce a
    single NIXL registration. When each tensor's ref goes out of scope via
    garbage_collect, the metadata_count decrements. After both are freed,
    the registration is removed.

    测试共享相同底层 storage 的两个重叠 tensor 只产生一个 NIXL 注册。
    当每个 tensor 的引用通过 garbage_collect 超出作用域时，metadata_count 递减。
    两者都释放后，注册被移除。"""
    from ray.experimental.rdt.nixl_tensor_transport import (
        NixlTensorTransport,
    )

    transport = NixlTensorTransport()

    tensor = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float32).to("cuda")
    view0 = tensor[0:2]
    view1 = tensor[1:3]
    storage_key = tensor.untyped_storage().data_ptr()

    assert view0.untyped_storage().data_ptr() == storage_key
    assert view1.untyped_storage().data_ptr() == storage_key
    assert view0.data_ptr() != view1.data_ptr()

    # Simulate ray.put(view0)
    # 模拟 ray.put(view0)
    obj_id1 = "test_obj_id_1"
    meta1 = transport.extract_tensor_transport_metadata(obj_id1, [view0])
    assert len(transport._tensor_desc_cache) == 1
    assert transport._tensor_desc_cache[storage_key].metadata_count == 1

    # Simulate ray.put(view1) and check that the a new entry is not created in the tensor desc cache
    # since they share the same storage key and the metadata_count is incremented by 1
    # 模拟 ray.put(view1)，并检查在 tensor desc cache 中不会创建新条目，
    # 因为它们共享相同的 storage key，且 metadata_count 增加 1
    obj_id2 = "test_obj_id_2"
    meta2 = transport.extract_tensor_transport_metadata(obj_id2, [view1])
    assert len(transport._tensor_desc_cache) == 1
    assert transport._tensor_desc_cache[storage_key].metadata_count == 2

    # Simulate the obj ref for view0 going out of scope and check that the nixl memory registration is
    # not cleared since the object ref for view1 is still in scope
    # 模拟 view0 的对象引用超出作用域，并检查 NIXL 内存注册未被清除，
    # 因为 view1 的对象引用仍在作用域内
    transport.garbage_collect(obj_id1, meta1, [view0])
    assert storage_key in transport._tensor_desc_cache
    assert transport._tensor_desc_cache[storage_key].metadata_count == 1

    # Simulate the obj ref for view1 going out of scope and check that the nixl memory registration is cleared
    # 模拟 view1 的对象引用超出作用域，并检查 NIXL 内存注册已被清除
    transport.garbage_collect(obj_id2, meta2, [view1])
    assert storage_key not in transport._tensor_desc_cache


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class OverlappingViewProducer:
    def produce_overlapping_views(self):
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to("cuda")
        slices = [tensor[0:2], tensor[1:3], tensor[2:4]]
        refs = []
        for s in slices:
            refs.append(ray.put(s, _tensor_transport="nixl"))
        return refs


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_storage_level_overlapping_views(ray_start_regular):
    """Test that overlapping views of the same storage tensor are properly transferred.

    测试同一 storage tensor 的重叠 view 是否被正确传输。"""

    actors = [OverlappingViewProducer.remote(), GPUTestActor.remote()]
    src_actor, dst_actor = actors[0], actors[1]

    refs = ray.get(src_actor.produce_overlapping_views.remote())
    result = ray.get(dst_actor.consume_with_nixl.remote(refs))
    assert result == 15


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class WaitTensorFreedActor:
    def test_wait_tensor_freed_views(self):
        from ray.experimental import wait_tensor_freed

        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to("cuda")
        slices = [tensor[0:3], tensor[1:4], tensor[2:5]]
        ref1 = ray.put(slices[0], _tensor_transport="nixl")
        ref2 = ray.put(slices[1], _tensor_transport="nixl")
        ref3 = ray.put(slices[2], _tensor_transport="nixl")
        del ref1
        wait_tensor_freed(slices[0], timeout=10)
        with pytest.raises(TimeoutError):
            wait_tensor_freed(slices[1], timeout=1)
        with pytest.raises(TimeoutError):
            wait_tensor_freed(slices[2], timeout=1)
        del ref2
        with pytest.raises(TimeoutError):
            wait_tensor_freed(slices[2], timeout=1)
        wait_tensor_freed(slices[1], timeout=10)
        del ref3
        wait_tensor_freed(slices[2], timeout=10)
        return "Success"


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_wait_tensor_freed_views(ray_start_regular):
    """Test that wait_tensor_freed tracks each view independently,
    not the shared underlying storage.

    测试 wait_tensor_freed 独立跟踪每个 view，
    而不是共享的底层 storage。"""
    actor = WaitTensorFreedActor.remote()
    result = ray.get(actor.test_wait_tensor_freed_views.remote())
    assert result == "Success"


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_get_into_tensor_buffers(ray_start_regular):
    @ray.remote(num_gpus=1, num_cpus=0)
    class GPUTestActor:
        def __init__(self):
            self.tensor_list = [
                torch.tensor([1, 2, 3]).to("cuda"),
                torch.tensor([4, 5, 6]).to("cuda"),
            ]

        def get_ref(self):
            return ray.put(self.tensor_list, _tensor_transport="nixl")

        def get_with_buffers(self, refs):
            set_target_for_ref(refs[0], self.tensor_list)
            tensors = ray.get(refs[0])
            # Make sure we ray.get-ted into the buffers
            # 确保 ray.get 操作写入了预分配的 buffer
            for new_tensor, tensor_buffer in zip(tensors, self.tensor_list):
                assert id(new_tensor) == id(tensor_buffer)
            return True

        def get_with_wrong_buffers(self, refs):
            wrong_tensor_buffer = [
                torch.tensor([1, 2]).to("cuda"),
                torch.tensor([4, 5]).to("cuda"),
            ]
            set_target_for_ref(refs[0], wrong_tensor_buffer)
            with pytest.raises(ValueError) as excinfo:
                ray.get(refs[0])
            assert "Shape of tensor_buffer at index 0" in str(excinfo.value)
            return True

    actors = [GPUTestActor.remote() for _ in range(2)]
    ref = ray.get(actors[0].get_ref.remote())
    result = actors[1].get_with_buffers.remote([ref])
    assert ray.get(result)

    result = actors[1].get_with_wrong_buffers.remote([ref])
    assert ray.get(result)


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_register_deregister_nixl_memory(ray_start_regular):
    """
    Test that register_nixl_memory persists the NIXL memory registration when the object ref goes out of scope

    测试 register_nixl_memory 在对象引用超出作用域时保持 NIXL 内存注册
    """
    from ray.experimental.rdt.nixl_tensor_transport import (
        NixlTensorTransport,
    )

    transport = NixlTensorTransport()
    tensor = torch.tensor([1, 2, 3]).to("cuda")

    transport.register_nixl_memory(tensor)
    key = tensor.untyped_storage().data_ptr()
    assert key in transport._tensor_desc_cache
    assert transport._tensor_desc_cache[key].metadata_count == 1

    # Simulate ray.put via extract_tensor_transport_metadata and bump the reference count
    # 模拟通过 extract_tensor_transport_metadata 执行 ray.put 并增加引用计数
    obj_id = "test_obj_id"
    meta = transport.extract_tensor_transport_metadata(obj_id, [tensor])
    assert transport._tensor_desc_cache[key].metadata_count == 2

    # Simulate GC via garbage_collect and decrement the reference count
    # 模拟通过 garbage_collect 执行 GC 并减少引用计数
    transport.garbage_collect(obj_id, meta, [tensor])
    assert key in transport._tensor_desc_cache
    # The reference count should be 1 due to being bumped by register_nixl_memory
    # 引用计数应为 1，因为被 register_nixl_memory 增加过
    assert transport._tensor_desc_cache[key].metadata_count == 1

    # decrement the remaining count to 0 and deregister the memory
    # 将剩余计数减至 0 并注销内存注册
    transport.deregister_nixl_memory(tensor)
    assert key not in transport._tensor_desc_cache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 2}], indirect=True)
def test_nixl_memory_pool(ray_start_regular, device):
    """
    Test NIXL memory pool: use the pre-allocated memory pool for NIXL transfers when available.
    When the pool cannot accommodate an allocation, an error is raised.

    测试 NIXL 内存池：当可用时，使用预分配的内存池进行 NIXL 传输。
    当内存池无法容纳一次分配时，将抛出错误。
    """

    @ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
    class PoolActor:
        def __init__(self, pool_device, pool_size):
            from ray.experimental import register_nixl_memory_pool

            register_nixl_memory_pool(pool_size, torch.device(pool_device))

        @ray.method(tensor_transport="nixl")
        def echo(self, data, device):
            return data.to(device)

        def get_num_managed_meta_nixl(self):
            return get_tensor_transport_manager("NIXL")._get_num_managed_meta_nixl()

    src_actor = PoolActor.remote(device, 48)
    dst_actor = GPUTestActor.remote()

    # Transfer the first small tensor (using memory pool internally).
    # 传输第一个小型 tensor（内部使用内存池）。
    ref1 = src_actor.echo.remote(torch.tensor([1, 2, 3]).to(device), device)
    assert ray.get(dst_actor.sum.remote(ref1, device)) == 6

    # Transfer the second small tensor (using memory pool internally).
    # 传输第二个小型 tensor（内部使用内存池）。
    ref2 = src_actor.echo.remote(torch.tensor([4, 5, 6]).to(device), device)
    assert ray.get(dst_actor.sum.remote(ref2, device)) == 15

    # Third transfer: pool is full. The allocation raises
    # NixlOutOfMemoryError, which surfaces as a RayTaskError.
    # 第三次传输：内存池已满。分配操作抛出 NixlOutOfMemoryError，
    # 该错误以 RayTaskError 的形式呈现。
    ref3 = src_actor.echo.remote(torch.tensor([7, 8, 9]).to(device), device)
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(dst_actor.sum.remote(ref3, device))
    assert "NixlOutOfMemoryError" in str(excinfo.value) and "out of memory" in str(
        excinfo.value
    )

    del ref1, ref2, ref3

    # Wait for GC to free the tensors on the sender.
    # 等待 GC 释放发送方上的 tensor。
    wait_for_condition(
        lambda: ray.get(src_actor.get_num_managed_meta_nixl.remote()) == 0,
        timeout=10,
        retry_interval_ms=100,
    )

    # Transfer the fourth tensor (after GC, using memory pool internally).
    # 传输第四个 tensor（GC 后，内部使用内存池）。
    ref4 = src_actor.echo.remote(torch.tensor([1, 2, 3, 4, 5, 6]).to(device), device)
    assert ray.get(dst_actor.sum.remote(ref4, device)) == 21


@pytest.mark.parametrize("ray_start_regular", [{"num_gpus": 1}], indirect=True)
def test_nixl_memory_pool_view_deduplication(ray_start_regular):
    """
    Test that views of the same tensor within a single ray.put share a single
    pool allocation, and that across ray.put calls the same storage reuses its
    pool slot.

    测试在同一 ray.put 中同一 tensor 的多个 view 共享一次内存池分配，
    以及在不同 ray.put 调用之间相同的 storage 复用其内存池槽位。
    """
    from ray.experimental.rdt.nixl_tensor_transport import (
        NixlTensorTransport,
    )

    transport = NixlTensorTransport()
    base = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32).to("cuda")
    storage_size = base.untyped_storage().nbytes()

    # Pool sized to exactly one full storage copy — enough for the shared
    # storage, and small enough that a duplicate allocation would fail.
    # 内存池大小恰好为一个完整 storage 副本——足够容纳共享的 storage，
    # 且足够小以至于重复分配会失败。
    transport.register_nixl_memory_pool(storage_size, torch.device("cuda"))

    view_a = base[0:2]
    view_b = base[1:3]

    # Both views share the same storage
    # 两个 view 共享相同的 storage
    assert view_a.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert view_b.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

    # Put both views in one object — shared storage should be allocated only once,
    # but metadata_count increments once per tensor.
    # 将两个 view 放入同一个对象——共享 storage 应只分配一次，
    # 但 metadata_count 每个 tensor 增加 1。
    obj_id1 = "view_obj_1"
    meta1 = transport.extract_tensor_transport_metadata(obj_id1, [view_a, view_b])
    ptr = base.untyped_storage().data_ptr()
    pool = transport._memory_pool
    assert pool.has_block(base)
    assert ptr in transport._tensor_desc_cache
    assert transport._tensor_desc_cache[ptr].reg_desc is None
    assert transport._tensor_desc_cache[ptr].metadata_count == 2

    # Second put of the same view — should reuse the same pool slot (cross-call cache)
    # 第二次 put 同一个 view——应复用相同的内存池槽位（跨调用缓存）
    obj_id2 = "view_obj_2"
    meta2 = transport.extract_tensor_transport_metadata(obj_id2, [view_a])
    assert pool.has_block(base)
    assert transport._tensor_desc_cache[ptr].metadata_count == 3

    # GC: metadata_count decrements once per tensor passed in, symmetric with
    # _add_pool_tensor_descs.
    # GC：metadata_count 每传入一个 tensor 减 1，与 _add_pool_tensor_descs 对称。
    transport.garbage_collect(obj_id1, meta1, [view_a, view_b])
    assert ptr in transport._tensor_desc_cache
    assert transport._tensor_desc_cache[ptr].metadata_count == 1

    transport.garbage_collect(obj_id2, meta2, [view_a])
    # All refs gone, pool block freed
    # 所有引用已清除，内存池块已释放
    assert ptr not in transport._tensor_desc_cache
    assert not pool.has_block(base)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
