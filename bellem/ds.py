# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/ds.ipynb.

# %% auto 0
__all__ = ['flatten_dict', 'unflatten_dict', 'NestedDict']

# %% ../nbs/ds.ipynb 4
def flatten_dict(d: dict, sep='.') -> dict:
    def recurse(subdict, parent_key=None):
        result = {}
        for k, v in subdict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                result.update(recurse(v, new_key))
            else:
                result[new_key] = v
        return result

    return recurse(d)

def unflatten_dict(d: dict, sep='.') -> dict:
    res = {}
    for k, v in d.items():
        subkeys = k.split(sep)
        container = res
        for subkey in subkeys[:-1]:
            if subkey not in container:
                container[subkey] = {}
            container = container[subkey]
        container[subkeys[-1]] = v
    return res

# %% ../nbs/ds.ipynb 7
class NestedDict(dict):
    def __init__(self, data, sep='.'):
        super().__init__(data)
        self.sep = sep
    
    def at(self, keys: str | list | tuple, default=None):
        if isinstance(keys, str):
            keys = keys.split(self.sep)
        node = self
        for key in keys:
            if key not in node:
                return default
            node = node.get(key)
        return node

    def set(self, keys: str | list | tuple, value):
        if isinstance(keys, str):
            keys = keys.split(self.sep)
        node = self
        last_key = keys.pop()
        for key in keys:
            if key not in node:
                node[key] = dict()
            node = node[key]
        node[last_key] = value

    def flat(self) -> dict:
        return flatten_dict(self, sep=self.sep)
    
    @classmethod
    def from_flat_dict(cls, data, sep='.'):
        return cls(unflatten_dict(data, sep=sep))
     
