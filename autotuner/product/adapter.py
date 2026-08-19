"""产品适配器协议与合同投影。

产品清单是输入，执行层和控制台只消费本模块生成的窄投影。这里不实现
训练、奖励或仿真逻辑，只校验入口身份和运行布局，避免调用方各自猜测
``runtime_package``、入口文件或远程目录。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import re
from typing import Any, Mapping, Protocol, runtime_checkable


ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
FILE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class ProductAdapterError(ValueError):
    """产品适配器声明不满足通用边界。"""


@runtime_checkable
class PayloadBuilder(Protocol):
    """产品 payload 构建器的最小协议。"""

    def __call__(self, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class LaunchPlanBuilder(Protocol):
    """可选的产品训练启动计划构建器协议。"""

    def __call__(self, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class SimulationAdapter(Protocol):
    """可选的产品仿真/sim2sim 适配器协议。"""

    def __call__(self, **kwargs: Any) -> Any:
        ...


def _contract_data(contract: Any) -> Mapping[str, Any]:
    if hasattr(contract, "to_dict") and callable(contract.to_dict):
        contract = contract.to_dict()
    if not isinstance(contract, Mapping):
        raise ProductAdapterError("product contract must be a mapping")
    return contract


def _safe_module(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not MODULE_RE.fullmatch(result):
        raise ProductAdapterError(f"{name} is not a safe module name: {result!r}")
    return result


def _safe_entrypoint(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not ENTRYPOINT_RE.fullmatch(result):
        raise ProductAdapterError(f"{name} must use module:callable form: {result!r}")
    return result


def _safe_file(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not FILE_RE.fullmatch(result):
        raise ProductAdapterError(f"{name} is not a safe Python filename: {result!r}")
    return result


def _safe_remote_path(value: Any, name: str, default: str) -> str:
    result = str(value or default).strip().rstrip("/")
    # 传输边界仍会做 shell 转义；这里先拒绝歧义路径，避免 glob 与审计记录漂移。
    if not result.startswith("/") or ".." in result.split("/") or not re.fullmatch(r"/[A-Za-z0-9_.:/-]+", result):
        raise ProductAdapterError(f"{name} is not a safe absolute remote path: {result!r}")
    return result


def _file_from_entrypoint(entrypoint: str) -> str:
    module = entrypoint.split(":", 1)[0]
    return module.rsplit(".", 1)[-1] + ".py"


@dataclass(frozen=True)
class ProductAdapterSpec:
    """供系统使用的产品适配器窄投影。"""

    product_id: str
    product_version: str
    contract_digest: str
    runtime_package: str
    payload_package: str
    payload_builder: str
    train_entrypoint: str
    diagnose_entrypoint: str
    payload_entrypoints: Mapping[str, str]
    deployment: Mapping[str, Any]
    simulation: Mapping[str, Any]
    asset_reuse: Mapping[str, Any]
    adaptation: Mapping[str, Any]
    plugins: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_contract(cls, contract: Any) -> "ProductAdapterSpec":
        data = _contract_data(contract)
        deployment = data.get("deployment") if isinstance(data.get("deployment"), Mapping) else {}
        training = data.get("training") if isinstance(data.get("training"), Mapping) else {}
        diagnostics = data.get("diagnostics") if isinstance(data.get("diagnostics"), Mapping) else {}
        declared_entries = deployment.get("payload_entrypoints")
        entries = dict(declared_entries) if isinstance(declared_entries, Mapping) else {}
        launcher = str(entries.get("launcher") or _file_from_entrypoint(training.get("train_entrypoint", "")))
        trainer = str(entries.get("trainer") or "train.py")
        diagnostic = str(entries.get("diagnostics") or _file_from_entrypoint(diagnostics.get("entrypoint", "")))
        package = str(deployment.get("runtime_package") or deployment.get("payload_package") or "").strip()
        payload_package = str(deployment.get("payload_package") or package).strip()
        result = cls(
            product_id=str(data.get("product_id") or "").strip(),
            product_version=str(data.get("product_version") or "").strip(),
            contract_digest=str(data.get("contract_digest") or "").strip(),
            runtime_package=package,
            payload_package=payload_package,
            payload_builder=str(deployment.get("payload_builder") or "").strip(),
            train_entrypoint=_safe_entrypoint(training.get("train_entrypoint"), "training.train_entrypoint"),
            diagnose_entrypoint=_safe_entrypoint(diagnostics.get("entrypoint"), "diagnostics.entrypoint"),
            payload_entrypoints={
                "launcher": launcher,
                "trainer": trainer,
                "diagnostics": diagnostic,
            },
            deployment=dict(deployment),
            simulation=dict(data.get("simulation")) if isinstance(data.get("simulation"), Mapping) else {},
            asset_reuse=dict(data.get("asset_reuse")) if isinstance(data.get("asset_reuse"), Mapping) else {},
            adaptation=dict(data.get("adaptation")) if isinstance(data.get("adaptation"), Mapping) else {},
            plugins={
                str(role): dict(value)
                for role, value in (data.get("plugins") or {}).items()
                if isinstance(value, Mapping)
            },
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not ID_RE.fullmatch(self.product_id):
            raise ProductAdapterError(f"invalid product_id: {self.product_id!r}")
        if not self.contract_digest:
            raise ProductAdapterError("product contract digest is required")
        _safe_module(self.runtime_package, "deployment.runtime_package")
        _safe_module(self.payload_package, "deployment.payload_package")
        _safe_entrypoint(self.payload_builder, "deployment.payload_builder")
        for role, filename in self.payload_entrypoints.items():
            _safe_file(filename, f"deployment.payload_entrypoints.{role}")
        _safe_remote_path(self.remote_root, "deployment.remote_root", "/root/gpufree-data/rl-agent")
        _safe_remote_path(self.runs_root, "deployment.runs_root", "/root/gpufree-data/runs")
        _safe_remote_path(self.data_root, "deployment.data_root", "/root/gpufree-data")
        runtime_python = str(self.deployment.get("runtime_python") or "").strip()
        if runtime_python:
            _safe_remote_path(runtime_python, "deployment.runtime_python", runtime_python)

    @property
    def remote_root(self) -> str:
        return _safe_remote_path(
            self.deployment.get("remote_root"),
            "deployment.remote_root",
            "/root/gpufree-data/rl-agent",
        )

    @property
    def legacy_payload_root(self) -> str:
        return _safe_remote_path(
            self.deployment.get("legacy_payload_root"),
            "deployment.legacy_payload_root",
            "/root/gpufree-data/training_payloads",
        )

    @property
    def runs_root(self) -> str:
        return _safe_remote_path(
            self.deployment.get("runs_root"),
            "deployment.runs_root",
            "/root/gpufree-data/runs",
        )

    @property
    def data_root(self) -> str:
        return _safe_remote_path(
            self.deployment.get("data_root"),
            "deployment.data_root",
            "/root/gpufree-data",
        )

    @property
    def run_prefix(self) -> str:
        value = str(self.deployment.get("run_prefix") or f"{self.product_id}_train").strip()
        if not ID_RE.fullmatch(value):
            raise ProductAdapterError(f"deployment.run_prefix is unsafe: {value!r}")
        return value

    @property
    def runtime_python(self) -> str:
        return str(self.deployment.get("runtime_python") or "python").strip()

    def process_pattern(self) -> str:
        """生成不会误匹配 grep/pkill 自身的产品训练进程正则。"""
        patterns: list[str] = []
        for module in self.process_modules():
            escaped = re.escape(module)
            patterns.append(f"[{re.escape(module[0])}]{escaped[len(re.escape(module[0])):]}")
        return "|".join(patterns)

    def payload_file(self, role: str) -> str:
        try:
            return self.payload_entrypoints[role]
        except KeyError as exc:
            raise ProductAdapterError(f"payload entrypoint role is not declared: {role}") from exc

    def process_modules(self) -> tuple[str, ...]:
        return tuple(
            f"{self.runtime_package}.{self.payload_file(role)[:-3]}"
            for role in ("launcher", "trainer")
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_product_adapter(contract: Any) -> ProductAdapterSpec:
    """将完整合同解析为可供控制台/执行层消费的产品适配器。"""
    return ProductAdapterSpec.from_contract(contract)


def load_plugin_entrypoint(reference: str, *, role: str = "plugin") -> Any:
    """按产品合同加载插件入口；这里只做加载，不自动执行。"""
    value = _safe_entrypoint(reference, role)
    module_name, attribute = value.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ProductAdapterError(f"{role} entrypoint is missing: {value}") from exc


__all__ = [
    "LaunchPlanBuilder",
    "PayloadBuilder",
    "ProductAdapterError",
    "ProductAdapterSpec",
    "SimulationAdapter",
    "load_plugin_entrypoint",
    "resolve_product_adapter",
]
