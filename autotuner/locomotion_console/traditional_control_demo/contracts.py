"""Stable contracts between the console and traditional-control providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from ..schemas import (
    DiagnosticPlayback,
    TraditionalControlDataSourceInfo,
    TraditionalControlProductInfo,
    TraditionalControlResult,
)


ProgressCallback = Callable[[float, str], None]
CancellationCallback = Callable[[], bool]


class DemoRunCancelled(RuntimeError):
    """A provider stopped cooperatively after the service requested cancellation."""


@dataclass(frozen=True)
class ProviderCatalog:
    """One provider's declarative UI catalog."""

    product: TraditionalControlProductInfo
    data_sources: tuple[TraditionalControlDataSourceInfo, ...]


@dataclass(frozen=True)
class ProviderRunResult:
    """Provider output before the service writes common artifacts."""

    result: TraditionalControlResult
    playback: DiagnosticPlayback
    raw_artifacts: Mapping[str, Path]
    provenance: Mapping[str, str] = field(default_factory=dict)


class TraditionalControlDemoProvider(Protocol):
    """Adapter boundary implemented by a product/simulator integration."""

    @property
    def provider_id(self) -> str: ...

    def catalog(self) -> ProviderCatalog: ...

    def run(
        self,
        *,
        controller_id: str,
        scene_id: str,
        data_source_id: str,
        duration_s: float,
        output_dir: Path,
        progress: ProgressCallback,
        cancelled: CancellationCallback,
    ) -> ProviderRunResult: ...

    def load_playback(self, output_dir: Path, max_frames: int = 900) -> DiagnosticPlayback: ...


__all__ = [
    "ProgressCallback",
    "CancellationCallback",
    "DemoRunCancelled",
    "ProviderCatalog",
    "ProviderRunResult",
    "TraditionalControlDemoProvider",
]
