# 历史归档文档

此目录包含历史参考文档，不作为当前系统的权威源。

## 用途

- **追溯历史决策**：查看过去的设计方案、失败教训、调参记录
- **比较方案**：对比不同时期的实现思路
- **查找已废弃的实现**：定位被替换或移除的代码逻辑

## 维护原则

- **只读**：归档文档不应被修改，保持历史完整性
- **不作为实现依据**：活跃开发应参考 `docs/` 根目录的当前文档
- **谨慎引用**：仅在必要时作为"历史证据"引用，不作为功能规格

## 活跃文档位置

当前系统的权威文档位于：

- `docs/README.md` - 文档索引
- `docs/repository_architecture.md` - 仓库架构
- `docs/atomic_problems.md` - 原子问题跟踪
- `docs/maintenance_standard.md` - 维护标准
- `README.md` - 项目入口

## 归档内容

### `taili/`

Taili 机器人项目的历史文档：

- `taili_session_lessons.md` - 1574 条用户指令的历史教训
- `taili_mujoco_investigation_20260803.md` - MuJoCo 调研发现
- `taili_tuning_followups.md` - 14 子系统对抗审计发现

这些文档记录了历史上的关键决策和失败案例，可作为参考，但不应作为当前实现的规格来源。
