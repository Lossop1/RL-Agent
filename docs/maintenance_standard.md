# 维护标准

## 修改前

1. 先判断改动属于机制、研究、基础设施、任务适配、控制台还是产物。
2. 先找权威入口和调用者；不要在兼容入口、payload 副本或历史备份上直接改业务逻辑。
3. 产品训练配置只编辑对应 `products/<product>/<task>/` 的权威配置，并确认合同中的 entrypoint 确实加载它；不要把产品字段复制到 `autotuner/`。
4. 记录影响范围；涉及训练、payload 或部署时，先保留当前运行和检查点，不覆盖用户已有改动。

## 修改中

- 新代码放在职责所属包；不要为了方便从 `locomotion_console` 导入研究或训练实现。
- 兼容模块只能转发到权威实现，不能定义业务类、业务函数或第二套常量。
- 新增注释使用中文，说明原因、边界和不变量；不要写日期化调参日记、无效历史猜测或重复代码内容。
- 复杂数学保留单位、参考系、窗口和门控语义；历史经验放在文档/台账，不复制成散落注释。
- 产品差异通过合同字段和插件角色表达；系统层只实现稳定协议，不能用机器人名称作为隐式分派条件。
- 不把 checkpoint、日志、诊断回放、payload 压缩包或前端 `dist` 当作源码提交。
- 远程执行必须经过已验证的基础设施适配，LLM 输入不能直接变成 shell 字符串。

## 验证顺序

以下命令必须串行执行，不能并行启动多个 PyTorch、IsaacLab 或训练进程：

```powershell
python tools/check_repository_structure.py
python -m compileall -q autotuner tests tools
python -m pytest tests/autotuner/<受影响包> -q
python -m products.<product>.payload.payload_manifest
python -m products.<product>.payload.build_payload --out output --stamp verify
cd locomotion-console-ui
npm.cmd run verify
```

训练、物理诊断和 MuJoCo 迁移属于运行验收，不能由上述静态检查替代；它们必须使用独立的
资源和明确记录的 payload/checkpoint。

## 文档规则

- 当前产品目标和验收语义写在对应产品合同与产品 spec；Taili 参考文件为 `docs/taili_spec.md`。
- 系统源码/运行边界写在本文档；产品源码映射写在对应产品目录文档。
- 操作步骤写在 `docs/taili_ops_runbook.md`、`docs/locomotion_console_startup.md` 和
  `docs/rl_agent_live_system_operations.md`。
- 设计演进、交接和历史调参记录放入 `docs/archive/`，不再作为当前权威入口。
- 文档入口统一从 `docs/README.md` 进入；删除或归档文档时必须更新反向链接。
