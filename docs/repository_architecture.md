# 仓库架构

## 目标

仓库同时承载训练运行时、研究编排、诊断验收、远程部署和操作界面。产品合同是系统与一次具体机器人任务之间的边界；`taili` 只是当前产品实例。整理后的核心原则是：

1. 每个职责只有一个可执行权威实现。
2. 研究逻辑不依赖 Web 控制台，训练核心不依赖 UI 或旧编排器。
3. 运行产物、历史证据和源码物理分区。
4. 兼容入口只能转发，不能重新实现业务逻辑。
5. payload 只能由清单从权威源码生成，远程副本不是源码。

## 分层

| 层 | 权威目录 | 允许承担的职责 |
|---|---|---|
| 机制 | `autotuner/mechanisms/` | 奖励、指标、门控的声明、校验、编译和安全运行时 |
| 研究 | `autotuner/research/` | 状态、台账、证据、候选、实验周期、监督和结果学习 |
| 基础设施 | `autotuner/infrastructure/` | SSH、进程和远程执行等可替换能力 |
| 执行层 | `autotuner/execution/` | runtime 身份、payload/run 哈希、ChangeSet、resume 判定、远程 staging 与原子激活 |
| 产品任务核心 | `products/<product>/core/` | 该产品的几何、观测、奖励数学、模型和课程纯逻辑 |
| 仿真/训练适配 | `products/<product>/<task>/` | 任务注册、环境、训练入口、运行遥测和物理验收适配 |
| 产品运维插件 | `products/<product>/ops/` | 产品特有的知识、playbook、验收和调参插件 |
| 控制台适配 | `autotuner/locomotion_console/` | FastAPI、数据源、UI API、LLM 工具和研究/远程适配 |
| 机器人适配 | `autotuner/adapter/` | 按产品合同消费机器人资产，生成通用部署计划；不保存某个机器人实现 |
| 诊断库 | `tools/isaaclab_quad_diag_observation/` | 可独立打包的诊断度量、记录和报告协议 |

执行层的唯一交付链是：

```text
config/products/<product>.yaml
    -> autotuner.product.ResolvedProductContract
    -> product/task payload builder
    -> autotuner.execution.DeploymentSpec
    -> runtimes/<runtime_digest> + payloads/<payload_digest> + runs/<run_id>
```

`config/products/<product>.yaml` 是产品合同入口，`products/<product>/` 是该产品
由机器人资产与任务要求得到的实现产物，`autotuner/` 是不依赖具体机器人的系统源码。
系统通过合同中的 entrypoint、插件角色和运行声明接入产品；不能因为当前只有 Taili
就把产品实现搬回系统目录。`output/`、`strategy_backups/` 和 `docs/archive/` 是运行
或历史产物，不能作为 Python import 源。

## 关键入口

- 产品合同：`config/products/<product>.yaml`
- 训练配置示例：`products/taili/blind_locomotion/taili_blind_config.yaml`（仅是 Taili 产品实现）
- 本地控制台：`python -m autotuner.locomotion_console`
- Taili 训练入口：`products/taili/blind_locomotion/launch_taili_train.py`
- payload 清单：`products/taili/payload/payload_manifest.py`
- payload 构建：`python -m products.taili.payload.build_payload`
- 结构门：`python tools/check_repository_structure.py`
- 前端源码：`locomotion-console-ui/`

## 依赖方向

依赖方向从底到顶：

`mechanisms -> research -> infrastructure / execution / product adapters -> console`

这不是严格的单链：产品核心可以依赖机制运行时，产品任务可以依赖产品核心，控制台
可以依赖研究层和基础设施。但以下反向依赖被禁止：机制/研究不能导入控制台；执行层
不能导入产品、任务或控制台；产品核心不能导入控制台或旧训练包；环境不能导入控制台。

## 源码与产物

| 类型 | 位置 | 处理规则 |
|---|---|---|
| 可审查源码 | `autotuner/`、`products/<product>/`、`tools/.../isaaclab_quad_diag/`、UI `src/public` | 进入版本控制，必须有测试或入口说明 |
| 测试源码 | `tests/` | 与权威包镜像，禁止写入源码目录 |
| payload 源映射 | `payload_manifest.py` | 唯一来源；不得手工维护第二份运行代码 |
| 运行输出 | `output/`、`frontend/dist/` | 可重建、默认忽略，不作为源码引用 |
| 历史资产 | `strategy_backups/`、归档文档 | 只用于追溯和比较，不参与 import/PYTHONPATH |
| 临时文件 | `.pytest-*`、`.pt`、`__pycache__`、`node_modules` | 不保留，清理后由忽略规则阻止回流 |
