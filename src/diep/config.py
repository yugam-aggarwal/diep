"""Global configuration variables for diep."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pymatgen.core.periodic_table import Element

# Default set of elements supported by universal diep models. Excludes radioactive and most artificial elements.
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.symbol not in ["Po", "At", "Rn", "Fr", "Ra"] and el.Z < 95)


# Default location of the cache for diep, e.g., for storing downloaded models.
DIEP_CACHE = Path(os.path.expanduser("~")) / ".cache/diep"
os.makedirs(DIEP_CACHE, exist_ok=True)

# Download url for pre-trained models.
PRETRAINED_MODELS_BASE_URL = "http://github.com/materialsvirtuallab/diep/raw/main/pretrained_models/"


def clear_cache(confirm: bool = True):
    """Deletes all files in the diep.cache. This is used to clean out downloaded models.

    Args:
        confirm: Whether to ask for confirmation. Default is True.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {DIEP_CACHE} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(DIEP_CACHE)
        except FileNotFoundError:
            print(f"diep cache dir {DIEP_CACHE!r} not found")
