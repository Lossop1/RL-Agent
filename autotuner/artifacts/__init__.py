"""训练、诊断和部署产物的通用索引。"""

from .training_archive import TrainingArchive, TrainingRunRecord
from .asset_catalog import (
    ASSET_CATALOG_SCHEMA,
    AssetBinding,
    AssetCatalog,
    AssetCatalogError,
    AssetCompatibilityReport,
    AssetLineage,
    AssetReuseApproval,
    ReusableAsset,
)

__all__ = [
    "ASSET_CATALOG_SCHEMA",
    "AssetBinding",
    "AssetCatalog",
    "AssetCatalogError",
    "AssetCompatibilityReport",
    "AssetLineage",
    "AssetReuseApproval",
    "ReusableAsset",
    "TrainingArchive",
    "TrainingRunRecord",
]
