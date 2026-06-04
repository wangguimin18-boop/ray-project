# RDT 英文注释中文翻译 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将所有 RDT 相关文件的英文注释翻译为中文，保留英文原文，在其后紧跟中文翻译。

**Architecture:** 逐文件读取、识别英文注释、在原注释后插入中文翻译行。使用并行子代理处理独立文件。

**Tech Stack:** Python/C++ 源文件，使用 opencode 的 edit/write 工具执行翻译。

---

## 翻译格式参考

### Python 单行注释
```python
# Before:
# This is an English comment

# After:
# This is an English comment
# 这是英文注释的中文翻译
```

### Python docstring
```python
# Before:
"""This is an English docstring describing the function."""

# After:
"""This is an English docstring describing the function.
这是描述该函数的英文文档字符串的中文翻译。"""
```

### C++ 单行注释
```cpp
// Before:
// This is an English comment

// After:
// This is an English comment
// 这是英文注释的中文翻译
```

### C++ 多行注释
```cpp
// Before:
/* This is a multi-line
   English comment */

// After:
/* This is a multi-line
   English comment */
/* 这是一个多行
   英文注释的中文翻译 */
```

### 规则
- 已含中文的注释跳过
- 技术术语保留英文（ObjectRef、Tensor、NCCL、RDMA 等）
- docstring 翻译追加在原 docstring 内，使用中文段落

---

## Task 1: __init__.py

**Files:**
- Modify: `python/ray/experimental/rdt/__init__.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/__init__.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/__init__.py
  git commit --signoff -m "翻译: experimental/rdt/__init__.py 英文注释添加中文翻译"
  ```

---

## Task 2: util.py

**Files:**
- Modify: `python/ray/experimental/rdt/util.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/util.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/util.py
  git commit --signoff -m "翻译: experimental/rdt/util.py 英文注释添加中文翻译"
  ```

---

## Task 3: rdt_store.py

**Files:**
- Modify: `python/ray/experimental/rdt/rdt_store.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/rdt_store.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/rdt_store.py
  git commit --signoff -m "翻译: experimental/rdt/rdt_store.py 英文注释添加中文翻译"
  ```

---

## Task 4: tensor_transport_manager.py

**Files:**
- Modify: `python/ray/experimental/rdt/tensor_transport_manager.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/tensor_transport_manager.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/tensor_transport_manager.py
  git commit --signoff -m "翻译: experimental/rdt/tensor_transport_manager.py 英文注释添加中文翻译"
  ```

---

## Task 5: collective_tensor_transport.py

**Files:**
- Modify: `python/ray/experimental/rdt/collective_tensor_transport.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/collective_tensor_transport.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/collective_tensor_transport.py
  git commit --signoff -m "翻译: experimental/rdt/collective_tensor_transport.py 英文注释添加中文翻译"
  ```

---

## Task 6: nixl_tensor_transport.py

**Files:**
- Modify: `python/ray/experimental/rdt/nixl_tensor_transport.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/nixl_tensor_transport.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/nixl_tensor_transport.py
  git commit --signoff -m "翻译: experimental/rdt/nixl_tensor_transport.py 英文注释添加中文翻译"
  ```

---

## Task 7: nixl_memory_pool.py

**Files:**
- Modify: `python/ray/experimental/rdt/nixl_memory_pool.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/nixl_memory_pool.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/nixl_memory_pool.py
  git commit --signoff -m "翻译: experimental/rdt/nixl_memory_pool.py 英文注释添加中文翻译"
  ```

---

## Task 8: cuda_ipc_transport.py

**Files:**
- Modify: `python/ray/experimental/rdt/cuda_ipc_transport.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/cuda_ipc_transport.py`

- [ ] **Step 2: 识别并翻译英文注释**
  识别所有英文注释和 docstring，在原注释后插入中文翻译

- [ ] **Step 3: 编辑文件**
  使用 Edit 工具逐步修改文件中的注释

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/cuda_ipc_transport.py
  git commit --signoff -m "翻译: experimental/rdt/cuda_ipc_transport.py 英文注释添加中文翻译"
  ```

---

## Task 9: rdt_manager.py (最大文件)

**Files:**
- Modify: `python/ray/experimental/rdt/rdt_manager.py`

- [ ] **Step 1: 读取文件内容**
  使用 Read 工具读取 `python/ray/experimental/rdt/rdt_manager.py`（注意文件较大，约958行，需要分段读取）

- [ ] **Step 2: 识别并翻译英文注释（分段进行）**
  分段识别所有英文注释和 docstring，在原注释后插入中文翻译。此文件注释最多，需要分段编辑。

- [ ] **Step 3: 编辑文件（分段编辑）**
  使用 Edit 工具分段修改文件中的注释，每次处理一段

- [ ] **Step 4: 验证翻译完整性**
  重新读取文件确认所有英文注释已翻译

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/experimental/rdt/rdt_manager.py
  git commit --signoff -m "翻译: experimental/rdt/rdt_manager.py 英文注释添加中文翻译"
  ```

---

## Task 10: worker.py (P1 — 部分翻译)

**Files:**
- Modify: `python/ray/_private/worker.py`（仅翻译 RDT 相关注释）

- [ ] **Step 1: 读取 RDT 相关部分**
  使用 Grep 搜索 worker.py 中包含 rdt_manager/rdt_store/direct_transport 的行，定位 RDT 相关代码段

- [ ] **Step 2: 读取相关代码段**
  读取 RDT 相关代码段附近的注释

- [ ] **Step 3: 翻译英文注释**
  仅翻译 RDT 相关的英文注释，其他注释不动

- [ ] **Step 4: 编辑文件**
  使用 Edit 工具修改 RDT 相关注释

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/_private/worker.py
  git commit --signoff -m "翻译: _private/worker.py RDT相关英文注释添加中文翻译"
  ```

---

## Task 11: serialization.py (P1 — 部分翻译)

**Files:**
- Modify: `python/ray/_private/serialization.py`（仅翻译 RDT 相关注释）

- [ ] **Step 1: 读取 RDT 相关部分**
  使用 Grep 搜索 serialization.py 中包含 rdt/RDT 的行，定位 RDT 相关代码段

- [ ] **Step 2: 读取相关代码段**
  读取 RDT 相关代码段附近的注释

- [ ] **Step 3: 翻译英文注释**
  仅翻译 RDT 相关的英文注释，其他注释不动

- [ ] **Step 4: 编辑文件**
  使用 Edit 工具修改 RDT 相关注释

- [ ] **Step 5: Commit**
  ```bash
  git add python/ray/_private/serialization.py
  git commit --signoff -m "翻译: _private/serialization.py RDT相关英文注释添加中文翻译"
  ```

---

## Task 12-19: C++ 核心文件 (P2)

每个 C++ 文件遵循相同流程：读取 → 翻译 → 编辑 → 验证 → Commit

- Task 12: `src/ray/common/ray_object.h`
- Task 13: `src/ray/core_worker/core_worker_options.h`
- Task 14: `src/ray/core_worker/task_manager.h`
- Task 15: `src/ray/core_worker/common.cc`
- Task 16: `src/ray/core_worker/task_manager.cc`
- Task 17: `src/ray/core_worker/core_worker.h`
- Task 18: `src/ray/core_worker/core_worker.cc`
- Task 19: `src/ray/core_worker/core_worker_process.cc`

---

## Task 20-26: 测试文件 (P3)

每个测试文件遵循相同流程：读取 → 翻译 → 编辑 → 验证 → Commit

- Task 20: `python/ray/tests/rdt/test_rdt_manager.py`
- Task 21: `python/ray/tests/rdt/test_rdt_gloo.py`
- Task 22: `python/ray/tests/rdt/test_rdt_ipc.py`
- Task 23: `python/ray/tests/rdt/test_rdt_nccl.py`
- Task 24: `python/ray/tests/rdt/test_rdt_nixl.py`
- Task 25: `python/ray/tests/rdt/test_rdt_custom.py`
- Task 26: `python/ray/tests/rdt/test_nixl_memory_pool.py`

---

## Task 27-29: 文档文件 (P4)

- Task 27: `doc/source/ray-core/direct-transport/rdt-architecture-analysis.md`
- Task 28: `doc/source/ray-core/direct-transport/direct-transport.rst`
- Task 29: `doc/source/ray-core/direct-transport/custom-tensor-transport.rst`

---

## 自审检查

1. **Spec覆盖**: 所有29个文件都已列出 ✓
2. **占位符扫描**: 无TBD/TODO ✓
3. **类型一致性**: 无代码类型需要检查（纯翻译任务）✓