# 研究运行闭环

本系统的研究主体不是某个机器人目录，而是一条可恢复的合同与证据谱系。产品
清单只提供资产、运行时和插件入口；任务合同编译器再根据用户任务生成训练、
遥测、诊断、仿真和部署五类 artifact。

## 运行顺序

1. 研究协调器接收结构化问题和受限证据摘要。LLM 只能产生结构化候选，不能
   产生 Python、Shell、SSH 或远程执行动作。
2. 确定性机制校验和人工批准通过后，`revise_and_prepare()` 创建新的不可变合同
   版本，并物料化五类 artifact、payload、运行 manifest 和部署规格。旧版本不被
   覆盖；manifest 记录父合同、证据和 resume 谱系。
3. 资产目录先登记内容摘要和来源。训练基线、checkpoint 等状态资产必须有来源
   合同和来源运行；复用还需要兼容性、能力范围、验证证据和针对目标合同/运行的
   一次性批准。批准的资产绑定会进入运行 manifest。
4. 只有 `VersionedRemoteDeployer` 激活成功后，协调器才把候选运行设为 active，
   但此时仍是 `deployed`，不代表训练进程已启动。远程部署回执写入 ledger，
   不保存认证信息。
5. 证据以 `EvidenceRecord` 导入并绑定到运行。评估提交必须引用 ledger 中已有、
   且属于同一运行的证据。
6. 裁决是确定性的：所有必需 evaluator 和受保护能力通过才 `promote`；证据不全
   或可恢复失败是 `continue`；受保护能力回归、终止性失败等触发 `rollback`。
   远程不可用时保留 `rollback_required`，恢复后可执行待处理回滚。

## 恢复原则

上下文丢失或进程重启后，先从 `ResearchStateStore` 恢复当前状态，再从 ledger
按 `research_run:<run_id>`、`deployment:<run_id>:...`、`evidence:<id>` 查询细节。
不要依据最近修改时间、checkpoint 文件名或聊天记忆猜测当前运行。合同、payload、
运行 manifest、资产记录和批准记录都保持内容摘要与追加式事件链。

## 产品扩展边界

Taili、blind 或未来的其他机器人型号只实现产品插件和各自清单；系统核心只消费
通用清单、适配器和合同。IsaacLab、MuJoCo、真机适配器以及具体奖励/诊断语义都
属于产品或任务产物，不能通过导入具体产品模块来改变协调器的安全边界。

## 部署与启动边界

`deploy_prepared()` 只激活经过校验的 runtime、payload 和运行 manifest。部署成功后
运行状态为 `deployed`；这不证明训练进程已经存在，也不会直接进入观察状态。

manifest 含产品生成的 `launch_plan` 时，`start_deployed()` 才把计划交给受限启动器。
启动器拒绝复用已存在的训练 tmux 会话，写入启动 marker，并验证声明的训练进程；
启动计划在准备阶段写入运行生命周期摘要；启动前摘要、argv、environment 和运行身份
必须仍然一致。这些检查全部成功后，协调器才追加 `TrainingStartReceipt` 并把运行升级为
`observing`。
启动失败只追加失败回执，保留 `deployed` 状态和 active run，以便从同一运行重试。

没有 `launch_plan` 的运行保持 deployment-only，协调器不会把它表示为已经验证启动。
部署回执、启动回执与生命周期分别追加保存，协调器重启后可从状态和 ledger 恢复待启动运行。
产品中立的 `VersionedTrainingStartExecutor` 只负责把 SSH 连接适配给执行层启动器；
它不拼接训练命令，也不复用会杀死已有 tmux 会话的旧启动入口。

启动回执先于观察状态形成；若本地状态 CAS 在远程启动后发生竞争，协调器会在仍指向
同一运行时重放状态落盘。后续重试会复用同一成功回执，不会再次创建训练会话；若 active
运行已经被替换，则只保留回执并拒绝自动重新激活旧运行，等待显式协调。
