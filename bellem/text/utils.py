# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/text.utils.ipynb.

# %% auto 0
__all__ = ['similarity', 'fuzzy_match']

# %% ../../nbs/text.utils.ipynb 3
from difflib import SequenceMatcher

# %% ../../nbs/text.utils.ipynb 4
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def fuzzy_match(a: str, b: str, threshold: float = 0.7):
    return similarity(a, b) >= threshold