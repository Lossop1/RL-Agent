# 任务合同产物链

这条链定义了“用户任务进入系统后，怎样得到可训练、可监控、可诊断、可仿真、可部署的产物”。它不把 Taili 或任何一个机器人型号写进系统核心。

用户目标
  -> LLM/人工结构化为 TaskRequest
  -> TaskContractCompiler
  -> ResolvedTaskContract
  -> TaskContractBundle
  -> TaskContractStore + TaskBundleMaterializer
  -> payload / run manifest / research ledger

研究闭环在上述准备阶段之上由 `autotuner.research.ResearchCoordinator` 统一编排：

```text
证据 -> 受限 LLM/机制合成候选 -> 确定性校验 -> 人工批准
     -> revise_and_prepare -> 远程激活 -> 证据导入
     -> promote / continue / rollback
```

## 边界

- autotuner/product/ 只定义合同、版本仓库和通用物料化协议。
- config/products/<product>.yaml 声明机器人资产、任务入口、仿真世界、部署边界和可选插件。
- products/<product>/ 承担该产品的训练、奖励、诊断、payload 和产品专用物料化。
- autotuner/execution/ 只消费 runtime、payload、run 和合同引用，不导入产品实现。
- autotuner/research/ 记录运行快照和证据来源，不把遥测或 LLM 判断直接升级为事实。
- autotuner/artifacts/asset_catalog.py 记录内容寻址资产、来源谱系和一次性复用批准；
  它不根据文件名或目录自动授予 checkpoint/训练基线复用资格。

## 合同版本

TaskContractStore 使用 contract_id@contract_version 作为版本引用。每个版本目录不可覆盖，保存记录包含：

- request_digest、product_contract_digest、contract_digest、bundle_digest
- parent_ref 与 supersedes_ref
- draft、approved、rejected、superseded 状态
- 追加式、带前序 digest 的 index.jsonl

状态变化只追加事件，不修改已保存的合同文件。恢复训练或回滚时，应加载合同引用和 payload/runtime digest，而不是根据目录名猜测。

## 五类产物

TaskBundleMaterializer 固定生成：

- training
- telemetry
- diagnostics
- simulation
- deployment

每个 artifact 同时记录合同 digest、bundle digest、生成器版本、内容 digest 和文件 digest。没有产品物料化插件时只生成通用声明式 spec，不猜测框架字段；产品需要落到实际训练配置时，声明 plugins.materializer.task_bundle，插件返回五类产品 spec。

## payload 与运行

产品 payload builder 可以接收 task_bundle。Taili builder 在接收到 bundle 时将合同和五类 spec 一起封装；旧 builder 不接收时仍保持兼容，但调用方必须知道它没有封装任务 bundle。

DeploymentSpec 携带：

- task_contract_ref
- task_bundle_digest

执行层把它们写入 run manifest 的 execution 段。研究 ledger 从 run manifest 恢复这些字段，避免同一产品、不同任务合同之间发生谱系混淆。

## LLM 的权限

LLM 可以提出结构化任务意图和候选修改，但不能自行批准合同、改变产品所有字段、跳过 artifact 校验或直接执行远程训练。确定性编译、物料化、payload 验证、部署授权和研究证据规则仍由核心代码负责。
## 统一编排入口

`TaskExecutionPipeline.prepare()` 是从合同到执行交接的唯一产品无关准备入口：

1. 保存不可变合同版本，并保留 `parent_ref` / `supersedes_ref`。
2. 在独立 artifact 仓库物料化 training、telemetry、diagnostics、simulation、deployment 五类配置。
3. 调用产品声明的 payload builder，并校验 payload、runtime 和任务 bundle 的 digest。
4. 生成带任务合同引用、bundle digest、artifact manifest 逻辑引用和 resume edge 的运行 manifest。
5. 把合同、运行快照和 resume 证据导入 append-only research ledger。

artifact 的执行路径只用于当前机器读取；跨机器和历史查询使用 `task-artifact-manifest:*`、`task-artifact:*` 等逻辑引用及 digest。

`TaskExecutionPipeline.revise()` 只能接收带 evidence refs 的已批准变更，生成新合同版本并建立父子谱系。LLM 可以提出意图，但不能伪造批准、跳过确定性校验或直接启动远程训练。

需要进入下一次训练时使用 `TaskExecutionPipeline.revise_and_prepare()`，而不是只调用
`revise()`。它会在新版本上重新生成五类 artifact、payload、运行 manifest 和
`DeploymentSpec`；旧版本不覆盖。批准的资产绑定会写入 manifest，resume checkpoint
还必须在启动计划中携带同一 `checkpoint_asset_ref`，否则交接被拒绝。

`ResearchCoordinator` 的验收规则是确定性的：必需 evaluator 和受保护能力全部通过
才可 promote；证据不完整进入 continue；受保护能力回归或终止性失败进入 rollback。
远程回滚暂时不可用时，状态会保留 `rollback_required`，进程重启后可从 ledger 恢复。

部署与训练启动是两个独立边界：`deploy_prepared()` 成功后运行处于 `deployed`，
必须再由 `start_deployed()` 通过启动计划、tmux 会话、marker 和进程可见性校验，
运行才进入 `observing`。启动失败保留 `deployed`，允许从同一运行重试；证据和评估
不能绕过启动回执直接附着到尚未验证启动的运行。
