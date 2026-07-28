"""Backward-compatible imports for the renamed :mod:`basic_mixins` module."""

from .basic_mixins import CheckpointCfgMixin, LoggerMixin, RayCfgMixin, RedisCfgMixin, WandbCfgMixin

__all__ = [
    "CheckpointCfgMixin",
    "LoggerMixin",
    "RayCfgMixin",
    "RedisCfgMixin",
    "WandbCfgMixin",
]
