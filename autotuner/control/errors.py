"""控制核心的结构化失败契约。"""

from __future__ import annotations

from typing import Mapping


class ControlError(RuntimeError):
    """可由运行器归档、但不能在控制核心内静默吞掉的失败。"""

    def __init__(
        self,
        component: str,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.component = str(component)
        self.code = str(code)
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "component": self.component,
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class ControlSolveError(ControlError):
    """数值求解器或控制约束不可行。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__("wbc", code, message, details=details)
