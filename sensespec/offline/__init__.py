"""Offline target-data materialization and training support."""

from sensespec.offline.dataset import (
    OFFLINE_SCHEMA_VERSION,
    OfflineDataset,
    configure_offline_args,
)

__all__ = [
    "OFFLINE_SCHEMA_VERSION",
    "OfflineDataset",
    "configure_offline_args",
]
