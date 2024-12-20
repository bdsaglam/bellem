# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/dspy.module.cte.ipynb.

# %% auto 0
__all__ = ['log', 'JERX', 'QA', 'validate_triple_format', 'validate_number_of_triples', 'ConnectTheEntities']

# %% ../../../nbs/dspy.module.cte.ipynb 3
import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature

from ...logging import get_logger

log = get_logger(__name__)

# %% ../../../nbs/dspy.module.cte.ipynb 4
class JERX(dspy.Signature):
    """Extract the triples relevant to the question from the given context."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    triples: list[tuple[str, str, str]] = dspy.OutputField(desc="List of triples (subject, predicate, object)")


class QA(dspy.Signature):
    """Answer the question based on the given triples."""

    triples: str = dspy.InputField(desc="List of triples (subject, predicate, object)")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def validate_triple_format(triple):
    return len(triple) == 3

def validate_number_of_triples(triples, max_n_triples: int):
    if isinstance(triples, str):
        triples = triples.split("\n")
    return len(triples) < 8

class ConnectTheEntities(dspy.Module):
    def __init__(self, max_n_triples=8):
        super().__init__()
        self._jerx = dspy.Predict(JERX)
        self._qa = dspy.Predict(QA)
        self.max_n_triples = max_n_triples

    def forward(self, context, question):
        triple_list = self._jerx(context=context, question=question).triples
        dspy.Suggest(
            all(validate_triple_format(triple) for triple in triple_list),
            "Triples must be in the format of (subject, predicate, object)",
            target_module=self._jerx,
        )
        dspy.Suggest(
            validate_number_of_triples(triple_list, self.max_n_triples),
            f"There must be max {self.max_n_triples} triples",
            target_module=self._jerx,
        )
        if isinstance(triple_list, list):
            triples = "\n".join(";".join(triple) for triple in triple_list)
        elif isinstance(triple_list, str):
            triples = triple_list
        else:
            raise ValueError("Unexpected type for triples")

        pred = self._qa(triples=triples, question=question)
        return dspy.Prediction(triples=triples, answer=pred.answer)
