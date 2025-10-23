"""Prompt templates for article generation contracts."""
from __future__ import annotations

from importlib import resources
from typing import Final

_TEMPLATE_PACKAGE: Final[str] = __name__


def load_template(name: str) -> str:
    """Return the contents of the template identified by ``name``.

    Parameters
    ----------
    name:
        File name within the :mod:`prompt_templates` package.

    Raises
    ------
    FileNotFoundError
        If the requested template does not exist.
    """

    with resources.files(_TEMPLATE_PACKAGE).joinpath(name).open("r", encoding="utf-8") as stream:
        return stream.read()
