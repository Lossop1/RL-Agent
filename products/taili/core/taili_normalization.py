"""观测归一化 - RunningMeanStd实现 P6.3验收标准：95%观测值落在[-3,3]。

使用Welford在线算法计算running statistics，避免数值不稳定。支持：
- 在线更新均值/方差（每个batch更新全局统计量）
- 归一化和反归一化
- 检查点保存/加载
- 训练/评估模式切换（冻结统计量）
- 多环境并行的批量更新

使用示例：
    rms = RunningMeanStd(shape=(1407,), device="cuda")

    # 训练阶段：更新统计量
    rms.unfreeze()
    observations = env.get_obs()  # (N, 1407)
    rms.update(observations)
    normalized = rms.normalize(observations, clip_range=3.0)

    # 评估阶段：冻结统计量
    rms.freeze()
    normalized = rms.normalize(observations)

设计原则：
- 数值稳定：Welford算法避免catastrophic cancellation
- 设备一致性：所有张量保持在同一device上
- 状态持久化：通过state_dict()集成到checkpoint系统
- 多环境支持：批量更新时正确聚合统计量
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """运行时均值/标准差估计器，使用Welford在线算法。

    Welford算法公式（数值稳定）：
        count_new = count + batch_size
        mean_new = mean + sum(batch - mean) / count_new
        M2_new = M2 + sum((batch - mean_old) * (batch - mean_new))
        var = M2 / (count - 1)

    相比朴素算法 var = E[x²] - E[x]²，Welford避免了大数相减导致的精度损失。
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        epsilon: float = 1e-8,
        device: str | torch.device | None = None,
    ):
        """初始化RunningMeanStd。

        Args:
            shape: 观测形状（不含batch维度）
            epsilon: 归一化时加到标准差上避免除零
            device: 张量所在设备（None则使用CPU）
        """
        super().__init__()
        self.epsilon = epsilon

        # 注册为buffer确保自动进入state_dict和device转换
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64, device=device))

        # 冻结标志（评估模式下不更新统计量）
        self._frozen = False

    def update(self, batch: torch.Tensor) -> None:
        """使用Welford算法更新running statistics。

        Args:
            batch: 观测batch，形状 (N, *shape) 其中N是batch size

        Note:
            冻结状态下调用此方法无效（静默忽略）
        """
        if self._frozen:
            return

        # batch: (N, *shape) -> 沿batch维度计算统计量
        batch_mean = batch.mean(dim=0)  # (*shape,)
        batch_var = batch.var(dim=0, unbiased=False)  # (*shape,) 使用N而非N-1
        batch_count = batch.shape[0]

        # Welford算法更新
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # 更新均值: mean_new = (count * mean_old + batch_count * batch_mean) / total_count
        #          = mean_old + batch_count * (batch_mean - mean_old) / total_count
        new_mean = self.mean + delta * batch_count / total_count

        # 更新方差的M2项（sum of squared differences）
        # M2_new = M2_old + M2_batch + delta² * count_old * batch_count / total_count
        # 注意：self.var存储的是方差，需要乘以count恢复M2
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        # 更新buffer - 避免count=0时除零
        self.mean.copy_(new_mean)
        if total_count > 0:
            self.var.copy_(M2 / total_count)  # 使用N而非N-1（与batch_var一致）
        self.count += batch_count

    def normalize(self, x: torch.Tensor, clip_range: float = 3.0) -> torch.Tensor:
        """归一化观测到 ~N(0,1) 并clip到 [-clip_range, +clip_range]。

        Args:
            x: 原始观测，形状 (N, *shape) 或 (*shape,)
            clip_range: clip范围（标准差的倍数），默认3.0即99.7%置信区间

        Returns:
            归一化后的观测，形状与输入相同

        Note:
            clip_range=3.0 确保 95%+ 的观测值落在 [-3, 3] 区间（P6.3验收标准）
        """
        std = torch.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        return torch.clamp(normalized, -clip_range, clip_range)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化观测回原始尺度。

        Args:
            x: 归一化后的观测

        Returns:
            原始尺度的观测
        """
        std = torch.sqrt(self.var + self.epsilon)
        return x * std + self.mean

    def freeze(self) -> None:
        """冻结统计量更新（评估模式）。

        冻结后调用update()不会修改mean/var/count。
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """解冻统计量更新（训练模式）。

        解冻后调用update()会正常更新mean/var/count。
        """
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """返回当前是否冻结状态。"""
        return self._frozen


__all__ = ["RunningMeanStd"]
