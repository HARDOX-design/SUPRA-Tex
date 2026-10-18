"""Supra-Tex Image-to-Material Translation."""

__all__ = ["prepare_image"]


def __getattr__(name):
    if name == "prepare_image":
        from .preprocess import prepare_image
        return prepare_image
    raise AttributeError(name)
