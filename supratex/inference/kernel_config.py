"""Select measured sparse-convolution launch configurations on a known runtime.

This bootstrap runs before importing FlexGEMM. It avoids benchmarking candidate
kernels during the first inference; it does not retain models or input tensors.
Unknown devices, versions and unlisted convolution shapes keep autotuning.
"""
from __future__ import annotations

import importlib
from importlib import metadata
import os
import sys

_CONFIGURED: dict | None = None
_CACHE_OPTIONS = ('FLEX_GEMM_AUTOTUNE_CACHE_PATH', 'FLEX_GEMM_USE_AUTOTUNE_CACHE',
                  'FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE')
_VALIDATED = {'device': 'NVIDIA H800', 'capability': (9, 0), 'torch': '2.6.0+cu124',
              'cuda': '12.4', 'triton': '3.2.0', 'flex_gemm': '1.0.0'}


def _runtime_signature() -> dict | None:
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        versions = {name: metadata.version(name) for name in ('triton', 'flex_gemm')}
    except metadata.PackageNotFoundError:
        return None
    return {'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'torch': str(torch.__version__), 'cuda': torch.version.cuda, **versions}


def configure_sparse_kernels() -> dict:
    """Apply the validated H800 preset, respecting explicit cache configuration.

    Call before importing sparse models. A packaged mapping is loaded through
    FlexGEMM's public API without reading or writing a machine's default cache.
    Its import-time cache flags are disabled only for this process, and temporary
    environment overrides are restored. Missing keys still use normal autotuning
    in memory. The return value reports whether a preset was selected.
    """
    global _CONFIGURED
    if _CONFIGURED is not None:
        return dict(_CONFIGURED)

    def skipped(reason: str) -> dict:
        return {'enabled': False, 'reason': reason}

    if any(name in os.environ for name in _CACHE_OPTIONS):
        return skipped('explicit_flex_gemm_cache_configuration')
    if os.environ.get('SPARSE_CONV_BACKEND', 'flex_gemm') != 'flex_gemm':
        return skipped('different_sparse_backend')
    for name in ('supratex.modules.sparse.config', 'pixal3d.modules.sparse.config'):
        config = sys.modules.get(name)
        if config is not None and getattr(config, 'CONV', 'flex_gemm') != 'flex_gemm':
            return skipped('different_sparse_backend')
    if 'flex_gemm' in sys.modules:
        return skipped('flex_gemm_already_imported')
    if _runtime_signature() != _VALIDATED:
        return skipped('unvalidated_runtime')

    from ._kernel_config_h800 import CONFIG

    # FlexGEMM otherwise loads the user's default disk cache at import. The
    # preset must also work from read-only installations, so use its mapping API
    # instead of a packaged cache path that would require creating a lock file.
    overrides = {'FLEX_GEMM_USE_AUTOTUNE_CACHE': '0',
                 'FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE': '0'}
    try:
        os.environ.update(overrides)
        flex_gemm = importlib.import_module('flex_gemm')
    finally:
        for name in overrides:
            os.environ.pop(name, None)
    flex_gemm.utils.load_autotune_cache(CONFIG)
    _CONFIGURED = {'enabled': True, 'preset': 'h800_torch26_cuda124_triton32_flexgemm1',
                   'configured_entries': sum(len(value) for value in CONFIG['NVIDIA H800'].values())}
    return dict(_CONFIGURED)
