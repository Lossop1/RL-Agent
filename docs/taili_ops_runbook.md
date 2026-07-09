# Taili 运营手册（LLM 大脑的第一手运营记忆）

这是本项目用几十小时训练实战换来的**运营知识**。诊断任何"卡住/异常/什么状态"问题前，先查这里的已知模式，
再结合 `get_operations_state`（系统自我状态：自动化在跑什么、停滞分类器结论）作判断。

## 已知故障模式（按发生频率）

### 1. GPU 内核挂起（本项目最常见故障，已发生 10+ 次）
**签名**：train.log 步数冻结（>3 分钟不动）+ 训练进程仍存活（S 态）+ GPU 显示假负载（100% 或很低）
+ 无任何报错/Traceback。遥测停更（stale）只是**结果**，不是原因。
**根因**：租用 GPU 盒子的透明故障，与训练代码无关（同一代码多次零停滞跑完 18k/37k 全程）。有明显的
"不稳定时段"（一小时内连续 3 次）与"稳定时段"交替。
**正确处置**：杀掉挂起进程，从最新 checkpoint 重启训练 —— **auto_drive 自愈驾驶会自动做**（检测
~10 分钟冻结→自动重启，见"系统自动化"）。诊断时不要猜测"写日志线程死锁/磁盘 IO 阻塞"之类 —— 从未
发生过；模式匹配上面的签名即可下结论。
**曾经的变体**：w_torque_margin=2.2 曾造成**确定性**挂起（每次都在 step 4600，物理爆炸），已回滚并
在杠杆上限里写死 ≤1.8。

### 2. 容器内存 OOM
**签名**：进程直接消失（非冻结），dmesg 有 `Memory cgroup out of memory`。约 50GB RSS 触发。
**处置**：用 `--num_envs 1024`（2048 会在 ~16-24k 步撞墙）。已是默认。

### 3. SSH 瞬断
**签名**："Error reading SSH protocol banner"。训练不受影响（tmux 独立），重连即可。

## 训练体制事实（决策时必须知道）
- **策略在 ~18k 续训步达到峰值**，练更久 B4 对称性和 A2 yaw 反而变差（过训）。按 18k 一轮迭代。
- **1024 envs 是稳定配置**；checkpoint 每 5000 步存一次。
- **B2 滑移门是度量伪影的历史**：物理评测曾把球足滚动计为滑移；现已改接触点速度。训练端奖励也已对齐
  （slip_free_speed 0.05）。B2 剩余超标是真实滑移。
- **A2 yaw 尾部对权重加码免疫**（2.5→2.8→3.1 三次无效，训练越久越差）——结构性问题，别再建议加
  w_tracking_yaw。
- **D 地形门只卡摔倒率**（速度标准全过）；stairs 摔倒率最高（曾 18.8%）。楼梯专项修复
  （离散地形强制抬脚 0.22m、后退地形限速）已入代码。
- **E1/E2 推扰门已通过**（0 摔倒，0.16s/0.25s 恢复）。

## 读训练进度（不只是"卡没卡"）
回答"训练怎么样/还要多久/有没有进展"前，先读 `get_operations_state` 的 `training_progress` 块，别从 step 数字猜。
- **1.5M `--total-steps` 是启动上限，不是目标。** 每轮训练都是从一个已会走的 checkpoint 续训；在 **~18k 续训步停下做全覆盖复测**（练更久会过训退化 B4/A2）。所以"才几千步、离 1.5M 还远"是错误心智模型——看的是"离 18k 续训步还差多少"。
- **绝不建议"再跑几万步再看"**——那正好越过 18k 峰值进入过训。
- **续训 ≠ 从零开始。** `is_resume`/`resumed_from_checkpoint` 告诉你策略已有的能力；累计能力 = 来源 checkpoint + 本轮步数。
- **curriculum 阶段不进 checkpoint**：续训靠 `TAILI_INIT_PHASE` 恢复（auto_drive 现在会自动从正在跑的进程继承 payload+phase）。看到 `current_phase=phi0` 而来源是高阶 checkpoint = 阶段没恢复对，是异常，不是正常预热。
- **刚改过奖励/curriculum 或从 phi0 重启后**：前几千步 progress/slip 等指标会先降后升（重新预热），**不要据此判定"结构瓶颈"**。跑到该阶段稳定了再评估。
- `loaded_payload` 告诉你这个 run 实际加载哪份代码（含哪些修复）——不同 payload 是不同实验，别混谈。

## 系统自动化（"是谁在动训练机"）
- **auto_drive**（自愈驾驶）：轮询步数,冻结≥3 个周期即自动杀+从最新 checkpoint 重启,到目标步数停下。
  它重启时 run 名会变（bench_时间戳）——看到 run 换名+从某 agent_XXXX 续起 = 自愈发生了,不是异常。
- **campaign / produce_policy**（自主调参/产出管线）：测评→分析→调参→部署→训练→复测循环,
  日志在 /tmp/produce_policy.log、/tmp/campaign*.json。
- **最佳策略注册表**：/root/gpufree-data/taili_runs/BEST_CHECKPOINT.json（评测创新高自动更新）;
  诊断/评测用 `best` 别名即可选中。

## 当前基线（随评测自动更新,以注册表为准）
- **修正 benchmark 完整基线：10/26 门实例**（0706）。最佳 = dirclr `bench_20260706_081321/agent_18000`（下楼梯摔倒 9.4%）。
- 完整过程记录:docs/taili_tuning_followups.md（D3a–D3s,每个修复的证据链）。

## Benchmark 结构（26 门实例；你作为负责人必须知道）
- **D 地形门分上下**（0706 起）：`D[stairs]` = **下楼梯**（普通金字塔,出生塔顶→往下走,判 fwd_speed+摔倒率）；`D[stairs_up]` = **上楼梯**（倒金字塔坑,坑底出生→必须爬出,判 `climbed>=0.15m + 不摔`,因为爬升慢,用横向速度会误杀）。两者是**不同能力**,别混。
- **失败门→杠杆对应表**（0706 实测）：
  - **A3+C**（立正,耦合）: A3 卡 duty_min<0.95(站立时有脚抬)→ 提 `w_stand_contact`(0.8→1.8,stand-gated 不伤行走)。最高置信 +2。
  - **A2 yaw**(2门): 0.7→0.47 靠 **turn-gated imit 奖励(rew_imitate 按转向占比缩放)+ mixed 阶段纯 yaw 采样(x_axis_prob<1)**。不是权重免疫,是可训练的,但 0.47→0.15 仍难。
  - **B4 对称**(4门): L/R duty 不对称随速度增大(fwd03 0.042→fwd07 0.104)。等变已开(修不了),疑与 **yaw-drift 同根**(高速跑偏→一侧多承重)。
  - **D[stairs] 下楼梯**: 别过度激进抬脚(会牺牲下楼梯稳定,dirclr 9.4%→激进配方 28%)。
  - **D[stairs_up] 上楼梯**: foot_h4 感知 + 爬升奖励 + 上楼梯地形课程。真硬,盲爬 25cm 在 SOTA 边界,当前 0.008m。
  - B1/B2/F2/A1-back: 边界/真硬,后排。
- **foot_h4 是盲狗爬楼梯的关键感知**(0706 D3s): TerrainPerceiver v1 只监督 slope/roughness,台阶高度通道曾是零占位→盲狗对台阶高度是瞎的。现已接上 foot_h4(每脚下地面高度)监督。

## 本场血泪教训(0706)——不犯这些才算真主人
- **过训陷阱**: 策略在 **~18k 续训步**峰值,52k 会退化(B4 0/5,总分 10→9)。auto_drive 现在**到目标步必杀训练**(不只停监控)。绝不让 run 无人监控地跑过头。
- **分析必须用全轨迹**: 判"爬没爬"要按 **per-case 全程范围**(max terrain_h - min),**不能只切前 1/3**(那是接近段,会误判"没爬")。我曾因此把能爬的策略误判成爬不动,围着假问题调了三刀。
- **上楼梯用 physeval D[stairs_up] 门,别用诊断**: diagnose_taili 的多地形只可靠记录 case 0,且倒金字塔大平台(platform 3.0)让机器人在平台上打转测不到台阶。可靠的上楼梯测量是 `physeval --terrain stairs_up`(platform 1.5,坑底直面台阶)。
- **部署双副本坑**: payload 里 `acceptance_score.py` 有两份——包内 `taili_blind_runtime/` 和 **payload 根**。physeval `import acceptance_score`(PYTHONPATH=根)加载**根副本**。改评分函数要**两处都部署**,否则报 AttributeError。
- **OOM 并发**: 训练 + 诊断两个 Isaac 叠加破 **50GB cgroup 限**→OOM 杀掉一个。绝不并发跑;auto_drive 已加诊断感知(诊断在跑就 HOLD 重启,不抢内存)。
- **auto_drive 血统**: 它从**正在跑进程的 environ**锁定 payload+phase(不再换回古董 bench_ payload);checkpoint 不存 curriculum 阶段,续训必须 `TAILI_INIT_PHASE=3`。
- **方法论铁律**: **测量不可信时绝不调参**。先验证测量(全程分析/正确地形/正确判据),再对着可信、进分的指标调。这一场大量弯路都源于对着坏测量硬调。

## 本场血泪教训(0707)——架构级改动 + 调试方法
- **并行 num_envs=1 冒烟(最重要)**: 改了训练代码要冒烟验证时,**不用停训练**——GPU 24GB、训练才占 ~12GB,`--num_envs 4` 的冒烟只 ~8GB,能和训练**并行跑**。我曾错误地反复停训练做冒烟,浪费了正在跑的进度。旧手册"2个Isaac必OOM"是过度保守(那是2个**满配**训练);小 num_envs 冒烟不 OOM。
- **skrl 集成调试:读真实源码,别猜**: 改 skrl 内部(如多critic fork 它的 `_update`/`record_transition`)时,**先把盒子上真实的 `skrl/agents/torch/amp/amp.py` 读下来**核对属性名/内存布局,不要凭知识猜(`_tensors_names`→`tensors_names`、`motion_fetch_samples`→`collect_reference_motions`、`_current_next_states` 不存在——都是猜错的假设)。一 bug 一冒烟极慢;一次性对齐所有内部假设。
- **K输出价值 vs skrl 标量假设(多critic 的头号坑)**: skrl 的 `record_transition` 有 `rewards += discount*values*truncated`(time_limit_bootstrap),假设 value 是标量[N,1];K输出价值[N,K] 会崩("[N,1] doesn't match [N,K]"),且级联成 NaN/卡死。**MC 时禁 `_time_limit_bootstrap`**(它自己的分组GAE用next_values引导,不需要)。这一个根因曾伪装成 ~10 个不同 bug。
- **多critic 必须分离冲突目标**: 多critic 的意义是给每个目标**独立归一化的话语权**。若把物理上冲突的目标(**yaw转向 vs 线速度**)归进同一组('track'),共享 advantage 会**主动压制**弱势目标——MC 反而没用。yaw 必须独立成 'turn' 组。z-score 独立归一后,原始权重失衡(1.0 vs 2.0)也被吸收。
- **yaw 课程:可达 ramp,别塌缩也别全硬**: yaw progress plateau ~0.50。①旧课程涨阈值 vel_cur_up=0.85 不可达→上限塌缩到 0.3(练太易、出分布外);②"全范围"钉死 0.9→从头拿0分→**放弃 yaw**(tracking_yaw 14k峰值后退化)。正解=**可达易→难 ramp**:易起点0.5、地板0.45(覆盖 yaw04)、yaw专属涨阈值 0.40(可达)让上限随改善涨到 0.9。
- **烧长跑前必跑验证 workflow**: 全栈架构改动前的对抗验证 workflow 抓到 **4 个会白烧 10-20× GPU 的 blocker**(SE/软等变写进了没被装载的文件、patch 顺序让感知器冻结、MC step0 崩、安全阀误杀 phi0)。代码审查 + 并行小冒烟是烧 GPU 前的廉价保险。
- **auto_drive run-id 特定**: 它 `pgrep -f train_taili` 匹配**任何** train_taili——并行冒烟时误判"训练还活着",停滞的被 pin 运行没被自愈。现在 `pgrep -f "--run-id {DRIVE_RID}"` 只认被 pin 的运行。
