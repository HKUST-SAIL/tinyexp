"""Internal helpers; experiments normally use them through the mixins.

- ``log_utils.tiny_logger_setup``: loguru file+console logger (used by ``LoggerCfgMixin``).
- ``model_utils.update_ema``: in-place EMA update of a model copy.
- ``ray_utils``: placement groups, worker env vars and rendezvous for ``RayCfgMixin``.
- ``redis_utils``: Redis client/cluster managers behind ``RedisCfgMixin`` and ``tinyexp-run-with-redis``.
"""
