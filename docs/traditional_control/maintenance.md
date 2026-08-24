# 传统控制工程维护规则

这份规则定义传统控制源码、可复用配置和运行产物的边界。目录边界不是约定
俗成的建议，而是提交、归档、清理和远程执行时的验收条件。

## Git 纳管范围

以下内容属于可审查、可复用的工程源，必须进入 Git：

- `autotuner/control/`：机器人无关的控制契约、MPC、连续步态、接触估计、WBC、阻抗层、trace 和 manifest 实现；
- `products/<product>/traditional_control/`：具体机器人 profile、运动学、后端适配、场景组合和打包入口；
- `config/traditional_control/`：版本化的控制 schema、增益、场景和验收阈值；
- `tests/` 中对应的单元、契约、回放和适配测试；
- `tools/traditional_control/` 中可从源码重建或验证产物的工具；
- `docs/traditional_control/` 中的架构、接口、维护规则和验证记录；
- `requirements-traditional-control.txt` 及必要的项目级依赖声明。

提交前必须确认：源码和配置没有密钥、远程路径、机器私有参数或临时绝对路径；
产品适配不能复制一份通用控制律；测试不能依赖某次本地运行留下的文件。

## 不进入 Git 的内容

以下是运行生成物或中间产物，不得提交，也不能作为 Python 源码被导入：

- `output/`、`.runtime/`、`.pytest-tmp*/` 和产品的 build/cache 目录；
- JSON/JSONL trace、诊断回放、评测报告、运行日志、临时 payload 和 staging 树；
- 内容寻址 bundle 的压缩包、checkpoint、模型文件和前端构建目录；
- `__pycache__/`、`*.pyc`、`node_modules/`、测试缓存和编译中间文件；
- SSH、LLM、环境变量和机器私有配置，包括 `config/*.local.json`、`.env` 和密钥文件。

即使某个中间产物很大、很新或能够帮助当前调试，也不能因此变成源码。需要
长期保留的证据必须通过 manifest 登记，并按哈希归档；没有来源、用途或复现
价值的失败产物在验证完成后删除。

## 产物生命周期

每次构建或评测使用一个工作区内的运行目录：

```text
output/traditional_control/
  archive/<artifact_digest>/     # 稳定 bundle、内容 manifest
  runs/<run_digest>/             # 一次运行的结果、trace 和运行 manifest
.runtime/traditional_control/    # 可删除的短期调试产物
```

稳定 bundle 的身份只由源码、配置、URDF、依赖声明和显式元数据决定；归档名
使用 `artifact_digest`，manifest 记录每个输入文件的 SHA-256。运行 manifest
另外记录时间、Git revision、工作树状态、依赖版本、使用的 bundle 摘要和运行
产物摘要。这样可以从 Git revision + manifest 重新构建，并能把一次结果回放到
精确输入，而不是依赖某个目录当前恰好存在的文件。

临时产物的清理规则是：先保留与当前问题、最佳结果或失败原因直接相关的
manifest/trace，再删除无 manifest 的临时文件、重复回放和可重新生成的缓存；
清理只能作用于工作区规定的 `output/`、`.runtime/` 和缓存目录，不能删除
`autotuner/`、`products/`、`config/`、`tests/`、`tools/`、`docs/` 或工作区外
的文件。清理后必须重新运行结构检查和受影响测试。

## 远程边界

远程服务器只接收已经在本地通过导入、契约、固定回放和名义场景回归的、带
manifest 的 bundle。远程使用独立 session 和运行目录，不修改本地 Git 源码，
不把源码当作远程维护副本，不覆盖现有 RL payload、checkpoint 或训练运行。
远程结果下载回本地后归入对应的 `runs/<run_digest>/`，不能直接散落到源码树。

## 提交前检查

提交或交付前按串行顺序执行：

1. `python tools/check_repository_structure.py`；
2. `python -m compileall -q autotuner/control products/taili/traditional_control tools/traditional_control`；
3. 受影响的 pytest；
4. `git diff --check` 和 `git status --short --untracked-files=all`；
5. bundle 构建，并确认 artifact manifest 可由当前源码和配置再次生成。

只要有未解释的生成文件、未登记的输入、越界的产物或无法重建的 bundle，
就不能声称工程整理完成。
