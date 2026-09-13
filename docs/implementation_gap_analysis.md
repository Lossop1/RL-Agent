# P3.1 SSH会话池管理 - 实现缺口分析

**分析日期**：2026-09-13  
**当前状态**：未开始（需要优化）  
**优先级**：高

---

## 当前实现状态

### 已有基础设施

1. **ParamikoSSH 类** (`autotuner/adapter/remote_deploy.py`)
   - 惰性连接（首次调用时建连）
   - 连接复用（`_conn()` 检查现有连接是否活跃）
   - 支持 exec 和 put 操作
   - 有 `close()` 方法显式关闭

2. **使用模式**
   ```python
   # 模式1：单次使用后立即关闭
   ssh = from_ssh_json()
   try:
       result = execute(plan, ssh, confirm=True)
   finally:
       ssh.close()
   
   # 模式2：适配器包装
   transport = RemoteSSHTransportAdapter(ssh, legacy_stderr=False)
   # 使用后也是立即关闭
   ```

### 问题分析

#### 问题1：频繁建连
**现象**：每次远程操作都创建新的 SSH 连接
```python
# autotuner/adapter/__main__.py
ssh = from_ssh_json()  # 新连接
try:
    execute(plan, ssh)
finally:
    ssh.close()  # 立即关闭

# 下次操作再次创建
ssh = from_ssh_json()  # 又是新连接
```

**影响**：
- SSH 握手延迟（典型 200-500ms）
- 服务器连接数压力
- 无法满足"连接建立次数 < 5 次/小时"的验收标准

#### 问题2：无全局连接池
**现象**：不同模块各自创建连接
- `adapter/__main__.py`：自己创建
- `locomotion_console/`：各自创建
- `research/research_supervisor.py`：可能也创建

**影响**：
- 同一主机的多个并发操作无法共享连接
- 内存和文件描述符浪费

#### 问题3：无连接健康检查
**现象**：`_conn()` 只检查 `is_active()`，不检查实际可用性
```python
def _conn(self):
    if self._client is not None:
        tr = self._client.get_transport()
        if tr is not None and tr.is_active():
            return self._client  # 可能已经僵死但 is_active() 返回 True
```

**影响**：
- 网络抖动后连接僵死，但未被检测
- 后续操作超时而不是快速重连

#### 问题4：无租约机制
**现象**：并发操作可能同时使用同一连接
```python
# 线程A
ssh.exec("command1")

# 线程B（同时）
ssh.exec("command2")  # Paramiko 内部可能冲突
```

**影响**：
- 命令输出混淆
- 竞态条件

---

## 验收标准差距

### 验收标准：连接建立次数 < 5 次/小时

**当前状态**：远超标准
- 每次 `from_ssh_json()` 调用 = 1 次建连
- 典型工作流：部署 + 启动训练 + 查询状态 = 至少 3 次
- 控制台操作（查看日志、诊断）：每次 2-5 次
- **实际**：可能 **20-50 次/小时**

**需要**：
- 连接池复用
- 空闲连接保活（如 10 分钟）
- 全局单例管理

---

## 实现方案

### 方案1：简单全局池（推荐用于快速实现）

```python
# autotuner/adapter/ssh_pool.py
import threading
import time
from typing import Dict, Tuple

class SimpleSSHPool:
    def __init__(self, idle_timeout: float = 600.0):
        self._pool: Dict[str, Tuple[ParamikoSSH, float]] = {}
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout
    
    def get(self, host: str, port: int, user: str, **kwargs) -> ParamikoSSH:
        key = f"{user}@{host}:{port}"
        with self._lock:
            if key in self._pool:
                ssh, last_used = self._pool[key]
                if time.time() - last_used < self._idle_timeout:
                    # 健康检查
                    if self._is_healthy(ssh):
                        self._pool[key] = (ssh, time.time())
                        return ssh
                # 超时或不健康，关闭旧连接
                try:
                    ssh.close()
                except Exception:
                    pass
            
            # 创建新连接
            ssh = ParamikoSSH(host, port, user, **kwargs)
            self._pool[key] = (ssh, time.time())
            return ssh
    
    def _is_healthy(self, ssh: ParamikoSSH) -> bool:
        try:
            # 快速健康检查
            ssh.exec("echo 1", timeout=2)
            return True
        except Exception:
            return False
    
    def cleanup_idle(self):
        """定期清理超时连接"""
        with self._lock:
            now = time.time()
            to_remove = [
                key for key, (ssh, last_used) in self._pool.items()
                if now - last_used >= self._idle_timeout
            ]
            for key in to_remove:
                ssh, _ = self._pool.pop(key)
                try:
                    ssh.close()
                except Exception:
                    pass

_global_pool = SimpleSSHPool()

def get_ssh_connection(host: str, port: int = 22, user: str = "root", **kwargs) -> ParamikoSSH:
    return _global_pool.get(host, port, user, **kwargs)
```

**优点**：
- 实现简单（约 80 行代码）
- 立即减少 80% 以上的建连次数
- 向后兼容（现有代码可逐步迁移）

**缺点**：
- 无租约隔离（需要调用者注意并发）
- 无连接数上限

### 方案2：完整连接池（推荐用于生产）

增加以下特性：
1. **连接租约**：借出连接时加锁，归还时释放
2. **最大连接数限制**：每个主机最多 N 个连接
3. **连接预热**：启动时预创建连接
4. **优雅关闭**：进程退出时关闭所有连接

```python
class ConnectionLease:
    def __init__(self, ssh: ParamikoSSH, pool, key: str):
        self._ssh = ssh
        self._pool = pool
        self._key = key
    
    def __enter__(self):
        return self._ssh
    
    def __exit__(self, *args):
        self._pool.return_connection(self._key, self._ssh)

class ManagedSSHPool:
    def __init__(self, max_per_host: int = 3, idle_timeout: float = 600.0):
        self._max_per_host = max_per_host
        self._idle_timeout = idle_timeout
        self._available: Dict[str, List[Tuple[ParamikoSSH, float]]] = {}
        self._in_use: Dict[str, Set[ParamikoSSH]] = {}
        self._lock = threading.Lock()
    
    def lease(self, host: str, port: int, user: str, **kwargs) -> ConnectionLease:
        key = f"{user}@{host}:{port}"
        with self._lock:
            # 尝试复用空闲连接
            if key in self._available and self._available[key]:
                ssh, _ = self._available[key].pop()
                if self._is_healthy(ssh):
                    self._in_use.setdefault(key, set()).add(ssh)
                    return ConnectionLease(ssh, self, key)
                # 不健康，关闭
                try:
                    ssh.close()
                except Exception:
                    pass
            
            # 检查是否超过限制
            in_use_count = len(self._in_use.get(key, set()))
            if in_use_count >= self._max_per_host:
                raise RuntimeError(f"SSH connection pool exhausted for {key}")
            
            # 创建新连接
            ssh = ParamikoSSH(host, port, user, **kwargs)
            self._in_use.setdefault(key, set()).add(ssh)
            return ConnectionLease(ssh, self, key)
    
    def return_connection(self, key: str, ssh: ParamikoSSH):
        with self._lock:
            self._in_use.get(key, set()).discard(ssh)
            self._available.setdefault(key, []).append((ssh, time.time()))
```

**使用示例**：
```python
pool = ManagedSSHPool(max_per_host=3, idle_timeout=600)

with pool.lease("10.0.0.1", 22, "root", password="xxx") as ssh:
    result = ssh.exec("ls -la")
    ssh.put("local.txt", "/remote/path.txt")
# 连接自动归还池中
```

---

## 迁移计划

### 阶段1：实现简单全局池（1天）
1. 创建 `autotuner/adapter/ssh_pool.py`
2. 实现 `SimpleSSHPool` 类
3. 添加单元测试（模拟 SSH 连接）

### 阶段2：改造现有调用点（1天）
1. 修改 `from_ssh_json()` 使用池
2. 更新 `adapter/__main__.py`
3. 更新 `remote_executors.py`
4. 保持向后兼容（可选显式 `close()`）

### 阶段3：验证和监控（0.5天）
1. 添加连接统计（建连次数、复用率）
2. 日志记录连接池状态
3. 实际运行验证 < 5 次/小时

### 阶段4（可选）：完整连接池（1-2天）
1. 实现 `ManagedSSHPool`
2. 迁移高并发场景（并行实验调度）
3. 添加监控仪表盘

---

## 预期效果

### 性能提升
- **建连次数**：从 20-50 次/小时 → **2-4 次/小时**（超过验收标准）
- **操作延迟**：每次操作节省 200-500ms 建连时间
- **资源占用**：减少服务器连接数 80%

### 验收标准达成
- ✅ 连接建立次数 < 5 次/小时
- ✅ 连接健康检查和自动重连
- ✅ 连接租约避免并发冲突

---

## 风险和注意事项

### 风险1：Paramiko 线程安全
**描述**：Paramiko 的 `SSHClient` 不是线程安全的
**缓解**：
- 方案1：每个线程独立连接（简单池足够）
- 方案2：租约机制确保独占使用（完整池）

### 风险2：连接泄漏
**描述**：异常退出时连接未归还
**缓解**：
- 使用 context manager (`with` 语句)
- atexit 钩子清理全局池

### 风险3：长时间空闲超时
**描述**：SSH 服务器可能主动断开长时间空闲连接
**缓解**：
- 配置合理的 `idle_timeout`（建议 10 分钟）
- 健康检查失败时自动重连

---

## 结论

**实现P3.1的路径清晰**：
1. 当前基础设施已有连接复用机制，但使用模式导致频繁建连
2. 简单全局池即可满足验收标准，2天可完成
3. 完整连接池提供更好的并发控制，可作为后续优化

**建议行动**：
- 立即启动阶段1和2（2天）
- 验证达标后，根据实际并发需求决定是否需要阶段4

**优先级理由**：
- 影响所有远程操作的性能
- 实现成本低（2天）
- 收益明显（减少 80% 建连）
- 是其他优化（P4.3 训练恢复、P1.1 前端操作）的基础
