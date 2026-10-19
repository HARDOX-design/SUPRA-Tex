from .RMBG2 import RMBG2, RMBG2_MODEL_PATH


_LEGACY_CLASS_NAMES = {"BiRefNet": "RMBG2"}


def get_rembg_class(name: str):
    """Resolve old checkpoint class names to the canonical RMBG2 wrapper."""
    canonical_name = _LEGACY_CLASS_NAMES.get(name, name)
    try:
        return globals()[canonical_name]
    except KeyError as error:
        raise AttributeError(f"Unknown background-removal class: {name}") from error
