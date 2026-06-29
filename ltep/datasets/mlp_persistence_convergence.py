"""Backward-compatibility shim -> ltep.datasets._mlp_persistence.
Kept so existing flat imports keep working. New code should import ltep.datasets._mlp_persistence."""
import ltep.datasets._mlp_persistence as _src
globals().update({k: getattr(_src, k) for k in dir(_src) if not k.startswith("__")})
del _src
