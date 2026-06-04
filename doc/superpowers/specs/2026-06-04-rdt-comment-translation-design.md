# RDT 英文注释中文翻译设计

## 目标

将 Ray 项目中所有 RDT (Ray Direct Transport) 相关文件的英文注释翻译为中文，保留原英文注释，在其后紧跟中文翻译。

## 翻译格式

| 类型 | 原格式 | 翻译后格式 |
|------|--------|------------|
| Python `#` 单行 | `# English comment` | `# English comment` + `# 中文翻译` |
| Python docstring | `"""English doc"""` | 保留原文，下方新增中文翻译段落 |
| C++ `//` 单行 | `// English comment` | `// English comment` + `// 中文翻译` |
| C++ `/* */` 多行 | `/* English */` | 保留原文，下方新增 `/* 中文 */` |

## 规则

- 已含中文的注释跳过，不重复翻译
- 翻译力求准确、简洁，技术术语保留英文（如 ObjectRef、Tensor、NCCL 等）
- 每个文件修改后单独提交

## 文件清单与优先级

### P0 — 核心模块 (9 文件)

- `python/ray/experimental/rdt/__init__.py`
- `python/ray/experimental/rdt/rdt_manager.py`
- `python/ray/experimental/rdt/rdt_store.py`
- `python/ray/experimental/rdt/tensor_transport_manager.py`
- `python/ray/experimental/rdt/collective_tensor_transport.py`
- `python/ray/experimental/rdt/nixl_tensor_transport.py`
- `python/ray/experimental/rdt/nixl_memory_pool.py`
- `python/ray/experimental/rdt/cuda_ipc_transport.py`
- `python/ray/experimental/rdt/util.py`

### P1 — Python 集成 (2 文件，部分翻译)

- `python/ray/_private/worker.py` (RDT 相关部分)
- `python/ray/_private/serialization.py` (RDT 相关部分)

### P2 — C++ 核心 (~8 文件)

- `src/ray/core_worker/task_manager.h`
- `src/ray/core_worker/task_manager.cc`
- `src/ray/core_worker/core_worker.h`
- `src/ray/core_worker/core_worker.cc`
- `src/ray/core_worker/core_worker_process.cc`
- `src/ray/core_worker/core_worker_options.h`
- `src/ray/core_worker/common.cc`
- `src/ray/common/ray_object.h`

### P3 — 测试文件 (7 文件)

- `python/ray/tests/rdt/test_rdt_nixl.py`
- `python/ray/tests/rdt/test_rdt_nccl.py`
- `python/ray/tests/rdt/test_rdt_manager.py`
- `python/ray/tests/rdt/test_rdt_ipc.py`
- `python/ray/tests/rdt/test_rdt_gloo.py`
- `python/ray/tests/rdt/test_rdt_custom.py`
- `python/ray/tests/rdt/test_nixl_memory_pool.py`

### P4 — 文档 (3 文件)

- `doc/source/ray-core/direct-transport/rdt-architecture-analysis.md`
- `doc/source/ray-core/direct-transport/direct-transport.rst`
- `doc/source/ray-core/direct-transport/custom-tensor-transport.rst`

## 总计约 29 个文件

## 执行策略

- 按优先级 P0→P1→P2→P3→P4 顺序执行
- 使用并行子代理加速翻译（每个子代理翻译一个文件）
- 每个文件修改后触发 commit-after-edit 纪律 skill