# 机制层

这里是奖励、指标、门控和安全表达式的唯一权威实现。机制层不依赖 Web 控制台、
具体机器人、IsaacLab、SSH 或训练进程；运行时需要的安全解释器由 payload 清单显式复制。

修改机制模型后必须运行：

```powershell
python tools/check_repository_structure.py
python -m pytest tests/autotuner/mechanisms -q
```

