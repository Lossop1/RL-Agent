"""机器人产品清单与注册表。

这里是系统与具体机器人产品之间的边界。系统只依赖通用清单契约；
不同机器人型号通过各自的配置清单接入。
"""

from .manifest import AssetSpec, ProductManifest, ProductManifestError, RobotProfile, load_product_manifest
from .contracts import ContractResolutionError, ResolvedAsset, ResolvedProductContract, resolve_product_contract
from .task_contract import (
    BUNDLE_SCHEMA,
    COMPILER_VERSION,
    ResolvedTaskContract,
    TASK_CONTRACT_SCHEMA,
    TaskContractBundle,
    TaskContractCompiler,
    TaskContractError,
    TaskRequest,
    compile_task_bundle,
)
from .payload import ProductPayload, ProductPayloadError, build_product_payload, make_deployment_spec
from .adapter import (
    LaunchPlanBuilder,
    PayloadBuilder,
    ProductAdapterError,
    ProductAdapterSpec,
    SimulationAdapter,
    load_plugin_entrypoint,
    resolve_product_adapter,
)
from .plugins import ProductPluginError, call_product_plugin, load_product_plugin, plugin_reference
from .runtime import ProductRuntimeView, resolve_product_runtime
from .launch import (
    LAUNCH_PLAN_SCHEMA,
    SUPPORTED_LAUNCHER_PROTOCOL,
    TrainingLaunchError,
    TrainingLaunchPlan,
    TrainingLaunchRequest,
    build_training_kill_command,
    build_training_launch_plan,
)
from autotuner.execution.runtime import RuntimeIdentity, resolve_runtime_identity
from .registry import ProductRegistry, get_product, get_robot_profile, validate_product

__all__ = [
    "AssetSpec",
    "BUNDLE_SCHEMA",
    "COMPILER_VERSION",
    "ContractResolutionError",
    "ProductManifest",
    "ProductManifestError",
    "ProductPayload",
    "ProductPayloadError",
    "ProductPluginError",
    "ProductAdapterError",
    "ProductAdapterSpec",
    "ProductRegistry",
    "ProductRuntimeView",
    "RobotProfile",
    "ResolvedAsset",
    "ResolvedProductContract",
    "ResolvedTaskContract",
    "RuntimeIdentity",
    "TASK_CONTRACT_SCHEMA",
    "TaskContractBundle",
    "TaskContractCompiler",
    "TaskContractError",
    "TaskRequest",
    "compile_task_bundle",
    "build_product_payload",
    "call_product_plugin",
    "get_product",
    "get_robot_profile",
    "load_product_manifest",
    "resolve_product_contract",
    "resolve_product_runtime",
    "make_deployment_spec",
    "LaunchPlanBuilder",
    "LAUNCH_PLAN_SCHEMA",
    "PayloadBuilder",
    "SimulationAdapter",
    "SUPPORTED_LAUNCHER_PROTOCOL",
    "TrainingLaunchError",
    "TrainingLaunchPlan",
    "TrainingLaunchRequest",
    "build_training_kill_command",
    "build_training_launch_plan",
    "load_plugin_entrypoint",
    "load_product_plugin",
    "plugin_reference",
    "resolve_product_adapter",
    "resolve_runtime_identity",
    "validate_product",
]
