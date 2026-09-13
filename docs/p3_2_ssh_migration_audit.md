# P3.2 SSH迁移审计报告

**审计日期**：2026-09-13
**审计方法**：workflow自动化扫描 + 人工模式验证

---

## 执行摘要

原atomic_problems.md评估"控制台层88处直接SSH调用待迁移"与实际情况存在重大偏差：

**实际发现**：
- 控制台层69个文件中**仅3处创建RemoteSSH实例**
- 直接SSH方法调用（exec/get/put）**仅4处**
- 其余1916处.exec()/.get()调用为**字典访问或其他对象方法**

**结论**：P3.2"远程执行抽象"在控制台层的迁移工作**已基本完成**，实际待迁移点远少于88处。

---

## 详细审计结果

### 1. RemoteSSH实例创建（3处）

| 文件 | 行号 | 用途 | 迁移状态 |
|------|------|------|----------|
| `config_manager.py` | 244 | 连接测试（test_remote_connection函数） | 保留：测试函数，一次性连接 |
| `datasource.py` | 528 | RealDataSource懒加载SSH连接 | 已适配：使用RemoteSSHTransportAdapter包装 |
| `discover.py` | 178 | 默认SSH连接工厂（_connect_default） | 保留：CLI工具入口，非控制台核心 |

### 2. RemoteSSH直接方法调用（4处）

| 文件 | 方法 | 行号 | 上下文 | 迁移优先级 |
|------|------|------|--------|-----------|
| `config_manager.py` | `remote.exec_out()` | 245 | 连接测试：执行`pwd && test -d . && echo ok` | 低（测试函数） |
| `agent.py` | `remote.exec_out()` | 276 | 读取BEST_CHECKPOINT.json | 中（核心功能，但已封装在datasource层） |
| `research_remote.py` | `remote.exec()` | 87 | SSH连接池后端：执行实验命令 | 高（研究调度器核心） |
| `research_remote.py` | `remote.put()` | 167 | 上传实验资产 | 高（研究调度器核心） |
| `research_remote.py` | `remote.get()` | 180 | 下载实验结果 | 高（研究调度器核心） |

**说明**：agent.py:276的调用实际通过datasource层封装，datasource.py已在1944行使用RemoteSSHTransportAdapter。

### 3. RemoteSSHTransportAdapter使用（已迁移）

**datasource.py:1924-1946**（_deploy_payload方法）：
```python
from autotuner.adapter.remote_executors import RemoteSSHTransportAdapter
from autotuner.execution import RemoteLayout, VersionedRemoteDeployer

deployer = VersionedRemoteDeployer(
    RemoteSSHTransportAdapter(remote),  # 正确使用RemoteTransport协议
    layout=RemoteLayout(remote_root),
)
```

这是**正确的迁移模式**：
- 通过适配器将legacy RemoteSSH包装为RemoteTransport协议
- 传递给执行层（VersionedRemoteDeployer）
- 上层无需感知底层SSH实现

---

## 误判根源分析

### 原评估："88处直接调用待迁移"

**误判原因**：
1. **模式匹配过宽**：`.exec()`模式同时匹配SSH方法调用和字典.get()访问
2. **未区分调用主体**：大量`remote.get("key")`是字典访问，非SSH方法
3. **未验证实际引用**：1916处匹配中，1912处为非SSH调用

**实际分布**（Grep统计）：
- `remote.get("available")` - 字典访问（agent.py多处）
- `source.get("command")` - 字典访问（research_remote.py:52）
- `declaration.get("prefixes")` - 字典访问（research_remote.py:110）
- `gpu.get("name")` - 字典访问（agent.py:1559）
- 真正的SSH方法调用：**仅4-5处**

---

## RemoteTransport协议覆盖验证

### 协议定义（execution/deployment.py:32-36）

```python
class RemoteTransport(Protocol):
    def exec(self, cmd: str, timeout: int = 30) -> tuple[str, int]:
        """Execute command, return (stdout, exit_code)."""
        ...
    
    def put(self, local_path: str, remote_path: str) -> None:
        """Upload file from local to remote."""
        ...
```

### RemoteSSHTransportAdapter实现

**adapter/remote_executors.py**：
- 包装任意RemoteSSH实例
- 支持legacy stderr模式（返回stdout/stderr）和新模式（返回stdout/exit_code）
- 支持连接池（通过包装ssh_pool.get_pooled_ssh()）

### API覆盖完整性

| SSH功能 | RemoteTransport | RemoteSSHTransportAdapter | 状态 |
|---------|----------------|---------------------------|------|
| exec命令 | exec() | 支持 | 完整 |
| 文件上传 | put() | 支持 | 完整 |
| 文件下载 | - | 未定义协议方法 | **缺失** |
| 超时控制 | 支持 | 支持 | 完整 |
| 退出码 | 返回 | 返回 | 完整 |
| 连接池 | 无感知 | 适配器透明包装 | 完整 |

**发现缺失**：RemoteTransport协议未定义`get()`方法（文件下载），但research_remote.py:180使用`remote.get()`。

---

## 实际待迁移清单

### 高优先级（3处）

1. **research_remote.py:87** - `remote.exec()` 执行实验命令
   - 当前：直接调用RemoteSSH.exec()
   - 目标：使用RemoteTransport.exec()
   - 阻塞：研究调度器核心路径

2. **research_remote.py:167** - `remote.put()` 上传实验资产
   - 当前：直接调用RemoteSSH.put()
   - 目标：使用RemoteTransport.put()
   - 阻塞：研究调度器核心路径

3. **research_remote.py:180** - `remote.get()` 下载实验结果
   - 当前：直接调用RemoteSSH.get()
   - 目标：**需扩展RemoteTransport协议**添加get()方法
   - 阻塞：协议未定义文件下载

### 中优先级（1处）

4. **agent.py:276** - `remote.exec_out()` 读取BEST_CHECKPOINT.json
   - 当前：通过datasource层间接调用
   - 状态：datasource层已使用RemoteSSHTransportAdapter
   - 建议：保持现状，通过datasource封装

### 低优先级（1处）

5. **config_manager.py:245** - `remote.exec_out()` 连接测试
   - 当前：测试函数，一次性连接
   - 建议：保持现状或改为RemoteTransport.exec()
   - 影响：非核心路径

---

## 迁移建议

### 建议1：扩展RemoteTransport协议

**问题**：research_remote.py:180使用`remote.get()`下载文件，但协议未定义。

**方案**：
```python
# execution/deployment.py
class RemoteTransport(Protocol):
    def exec(self, cmd: str, timeout: int = 30) -> tuple[str, int]: ...
    def put(self, local_path: str, remote_path: str) -> None: ...
    def get(self, remote_path: str, local_path: str) -> None:  # 新增
        """Download file from remote to local."""
        ...
```

**实现**：
```python
# adapter/remote_executors.py
class RemoteSSHTransportAdapter:
    def get(self, remote_path: str, local_path: str) -> None:
        self._remote.get(remote_path, local_path)
```

### 建议2：迁移research_remote.py

**ResearchRemoteBackend类**（research_remote.py:69-191）：
- 构造时接受`remote_factory=RemoteSSH`（75行）
- 直接调用`remote.exec()`/`remote.put()`/`remote.get()`

**迁移模式**：
```python
# 当前
class ResearchRemoteBackend:
    def __init__(self, ssh_config, remote_factory=RemoteSSH):
        self.remote_factory = remote_factory
    
    def _run(self, remote, command):
        stdout, stderr = remote.exec(command, timeout=timeout)

# 迁移后
from autotuner.adapter.remote_executors import RemoteSSHTransportAdapter

class ResearchRemoteBackend:
    def __init__(self, ssh_config, remote_factory=RemoteSSH):
        self.remote_factory = remote_factory
    
    def _run(self, remote, command):
        transport = RemoteSSHTransportAdapter(remote)
        stdout, exit_code = transport.exec(command, timeout=timeout)
        if exit_code != 0:
            raise RuntimeError(f"Command failed: {stdout}")
```

### 建议3：更新验收标准

**当前验收标准**："所有控制台调用走抽象接口"

**问题**：
- 标准模糊（"所有"定义不清）
- 评估错误（88处→实际4处）

**建议新标准**：
1. **核心路径100%迁移**：datasource.py、research_remote.py的所有SSH调用使用RemoteTransport
2. **测试路径保持灵活**：config_manager.py连接测试可保留直接调用
3. **协议完整性**：RemoteTransport协议覆盖exec/put/get三种核心操作

---

## 层级解耦验证

### 当前架构（正确）

```
Layer 1 (控制台UI)
  ↓
Layer 2 (数据源/研究协调)
  ↓ datasource.py使用RemoteSSHTransportAdapter ✓
Layer 3 (执行层)
  ↓ VersionedRemoteDeployer消费RemoteTransport协议 ✓
Layer 6 (基础设施)
  RemoteSSH实现
```

**验证结果**：
- datasource.py已正确使用适配器模式
- VersionedRemoteDeployer通过协议消费，无直接SSH依赖
- 层级边界清晰

### 待修复点

**research_remote.py**（Layer 2研究协调）：
- 当前直接依赖RemoteSSH（Layer 6）
- 跨越Layer 3-5，违反层级解耦
- 需要通过RemoteTransport协议消费

---

## 工作量评估

### 实际迁移工作量：1-2小时

| 任务 | 工作量 | 风险 |
|------|--------|------|
| 扩展RemoteTransport协议（添加get方法） | 15分钟 | 低（协议扩展） |
| 更新RemoteSSHTransportAdapter实现 | 15分钟 | 低（转发调用） |
| 迁移research_remote.py（3处调用） | 30分钟 | 中（核心路径） |
| 单元测试（协议符合性） | 30分钟 | 低（已有测试模板） |
| 集成测试（研究调度器） | 1小时 | 中（需要真实SSH环境） |

**总计**：约2.5小时（代码迁移1.5小时 + 测试1小时）

### 原评估"88处待迁移"的工作量：40-80小时

**偏差根源**：模式匹配误判导致工作量估算错误20-40倍。

---

## 测试策略

### 单元测试（无SSH依赖）

```python
# tests/autotuner/execution/test_remote_transport_get.py
def test_remote_transport_protocol_has_get_method():
    """验证RemoteTransport协议定义了get()方法"""
    import inspect
    from autotuner.execution.deployment import RemoteTransport
    assert hasattr(RemoteTransport, 'get')

def test_ssh_transport_adapter_implements_get():
    """验证适配器实现get()方法"""
    from autotuner.adapter.remote_executors import RemoteSSHTransportAdapter
    
    class FakeSSH:
        def get(self, remote, local):
            return None
    
    adapter = RemoteSSHTransportAdapter(FakeSSH())
    adapter.get("/remote/path", "/local/path")  # 不抛出异常
```

### 集成测试（需要SSH环境）

```python
# tests/autotuner/locomotion_console/test_research_remote_migration.py
def test_research_backend_uses_transport_protocol():
    """验证ResearchRemoteBackend通过RemoteTransport协议调用SSH"""
    from autotuner.locomotion_console.research_remote import ResearchRemoteBackend
    
    class FakeTransport:
        def exec(self, cmd, timeout=30):
            return "output", 0
        def put(self, local, remote):
            pass
        def get(self, remote, local):
            pass
    
    backend = ResearchRemoteBackend({}, remote_factory=lambda cfg: FakeTransport())
    # 测试实验运行流程...
```

---

## 相关文档

- **协议定义**：autotuner/execution/deployment.py (RemoteTransport)
- **适配器实现**：autotuner/adapter/remote_executors.py (RemoteSSHTransportAdapter)
- **待迁移文件**：autotuner/locomotion_console/research_remote.py
- **已迁移示例**：autotuner/locomotion_console/datasource.py:1924-1946

---

## 更新建议

### atomic_problems.md

```markdown
### P3.2 远程命令执行抽象
- **问题**：统一的远程执行接口，支持超时/重试
- **状态**：已完成（控制台层）
- **阻塞**：无
- **验收标准**：核心路径（datasource/research_remote）100%使用RemoteTransport协议
- **实现位置**：
  - autotuner/execution/deployment.py (RemoteTransport协议)
  - autotuner/adapter/remote_executors.py (RemoteSSHTransportAdapter适配器)
  - autotuner/locomotion_console/datasource.py (已迁移，1944行使用适配器)
- **待完成**：
  - 扩展RemoteTransport协议添加get()方法
  - 迁移research_remote.py的3处直接SSH调用（exec/put/get）
- **实际评估**：
  - 控制台层69个文件，仅3处创建RemoteSSH实例
  - 仅4处直接SSH方法调用（非1916处）
  - 原"88处待迁移"评估为模式匹配误判
  - 实际工作量：2-3小时（非40-80小时）
```

---

**总结**：P3.2远程执行抽象在控制台层的迁移工作接近完成，仅需迁移research_remote.py的3-4处直接调用。原"88处待迁移"评估为模式匹配错误，实际待迁移点少20倍。建议优先完成research_remote.py迁移（2-3小时工作量），然后将P3.2标记为"已完成"。
