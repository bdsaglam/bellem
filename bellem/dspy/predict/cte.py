# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/dspy.predict.cte.ipynb.

# %% auto 0
__all__ = ['log', 'ConnectTheEntities']

# %% ../../../nbs/dspy.predict.cte.ipynb 3
import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature

from ...logging import get_logger

log = get_logger(__name__)

# %% ../../../nbs/dspy.predict.cte.ipynb 4
class ConnectTheEntities(Module):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__()

        self.activated = activated

        self.signature = signature = ensure_signature(signature)

        prefix = "Let's identify the relevant entity-relation-entity triples in the format of 'subj | relation | obj', .e.g Glenhis Hernández | birth place | Havana\nMarta Hernández Romero | mayor of| Havana\n"
        desc = "${triples}"
        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)

        # Add "triples" field to the output signature.
        extended_signature = signature.prepend("triples", rationale_type, type_=str)

        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature

    def forward(self, **kwargs):
        assert self.activated in [True, False]

        signature = kwargs.pop("new_signature", self._predict.extended_signature if self.activated else self.signature)
        return self._predict(signature=signature, **kwargs)

    @property
    def demos(self):
        return self._predict.demos

    @property
    def extended_signature(self):
        return self._predict.extended_signature
