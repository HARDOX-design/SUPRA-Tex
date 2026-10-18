"""SUPRA-Tex observation-guided completion."""
from . import samplers

def __getattr__(name):
    if name in {"build_completion_model", "sample_material_latents"}:
        from . import observation_guided_completion
        return getattr(observation_guided_completion, name)
    raise AttributeError(name)
