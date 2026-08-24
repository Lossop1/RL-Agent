# 传统控制可视化演示

传统控制演示是控制台中的独立工具。它不修改 `autotuner.control` 的控制算法，也不要求本机启动 MuJoCo viewer。运行时在本机无头执行产品声明的控制器和仿真后端，随后由浏览器中的通用 Three.js/URDF 查看器显示统一回放。

## 用户操作

1. 启动后端控制台和前端 Vite 服务。
2. 打开「传统控制」页。
3. 从目录选择产品、控制器、场景、时长和数据源。
4. 选择「本机无头运行」后启动演示；页面轮询 job 状态。
5. 运行完成后，查看结果摘要和 manifest 路径，在查看器中播放、暂停、拖动、调速、重置视角或跟随机身。
6. 选择「已有本地回放」可以重新加载输出根目录下已有 run，不重新执行仿真。

输出目录默认是 `output/traditional_control/<run_id>/`，也可用 `LOCOMOTION_TRADITIONAL_CONTROL_ROOT` 指定。每次 run 至少包含：

- `manifest.json`：请求、provider、状态、时间和 artifact 路径；
- `control_trace.jsonl`：控制周期输入、状态和接触信息；
- `result.json`：控制评估结果和失败原因；
- `playback.json`：浏览器通用回放帧、机器人资源和场景 primitive。

## 扩展边界

产品清单的 `plugins.traditional_control_demo.provider` 声明 provider 工厂，provider 返回产品目录并实现统一运行/回放协议。控制台只负责：

- 读取产品注册表和 provider catalog；
- 校验选择并管理本地 job 生命周期；
- 写入 manifest、结果和回放文件；
- 提供 `/traditional-control/*` API。

provider 负责把具体控制器、仿真器和 trace 转换成统一 `DiagnosticPlayback`。前端只消费 `robot`、`scene.terrain_primitives` 和状态帧，不识别 Taili、MuJoCo 或 IsaacLab 名称。新增机器人、场景、后端或 trace 格式时，应新增产品配置/provider/测试，不在服务、控制核心或查看器中增加产品分支。

## 当前 Taili provider

Taili 的配置位于 `config/traditional_control/taili_nominal.yaml`，由产品 simulation 声明引用。它调用已有 `run_mujoco_episode`，本机只做无头计算；浏览器承担渲染。该演示用于观察运动和诊断数据，不等同于控制验收结论，验收仍以 `result.json` 的指标和独立测试为准。

## API

- `GET /traditional-control/catalog`
- `POST /traditional-control/run`
- `GET /traditional-control/status`
- `GET /traditional-control/history`
- `GET /traditional-control/playback`
- `GET /traditional-control/result`
- `GET /traditional-control/manifest`
- `POST /traditional-control/cancel`

状态帧协议在 `autotuner/locomotion_console/schemas.py` 中定义；它同时兼容已有 IsaacLab 诊断回放。
