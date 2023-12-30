import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import kuzu
import typer
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from llama_index import Document, KnowledgeGraphIndex, ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.storage.storage_context import StorageContext
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
from phoenix.trace.schemas import Span
from pyvis.network import Network
from rich.console import Console

from bellek.kuzu import KuzuGraphStore
from bellek.ml.llm.obs import TraceRecorder
from bellek.utils import generate_time_id

err = Console(stderr=True).print

load_dotenv()

set_llm_cache(SQLiteCache(database_path="/tmp/langchain-cache.db"))

LLM_OBS_DIRECTORY = Path("/tmp/phoenix/thesis-kg-llm/kgcons/")


def make_service_context(model_type: str, trace_callback: Callable[[List[Span]], None]):
    if model_type == "llama2-sft":
        from bellek.ml.llama_index import HuggingFaceTextGenInferenceLLM

        inference_server_url = "http://localhost:8080/"
        llm = HuggingFaceTextGenInferenceLLM(
            inference_server_url=inference_server_url,
            do_sample=False,
            max_new_tokens=200,
            repetition_penalty=1.0,
            # top_k=50,
            # top_p=1.0,
            # typical_p=1.0,
            # temperature=0.0,
        )
    else:
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

    # model to generate embeddings for triplets
    embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # Setup LLM observability
    traces_path = LLM_OBS_DIRECTORY / f"{model_type}" / f"traces-{generate_time_id()}.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    callback_manager = CallbackManager(handlers=[OpenInferenceTraceCallbackHandler(trace_callback)])

    return ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        callback_manager=callback_manager,
    )


LLAMA2_KG_TRIPLET_EXTRACT_TMPL = """<s>[INST] <<SYS>>
You are a helpful assistant that extracts up to {max_knowledge_triplets}  entity-relation-entity triplets from given text. Use '|' as delimiter and provide one triplet per line.
<</SYS>>
Alaa Abdul Zahra plays for Al Shorta SC. His club is AL Kharaitiyat SC, which has its ground at, Al Khor. [/INST] Al Kharaitiyat SC|ground|Al Khor
Alaa Abdul-Zahra|club|Al Kharaitiyat SC
Alaa Abdul-Zahra|club|Al Shorta SC </s><s>[INST] {text} [/INST] """

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = """
Some text is provided below. Given the text, extract up to {max_knowledge_triplets}  knowledge triplets in the form of (subject, predicate, object) that might be relevant to the following question. The subject and object must be different.
Prioritize triplets that:
1. Offer temporal information like 'founded in,' 'created on,' 'abolished in,' etc.
2. Provide spatial details such as 'located in,' 'borders,' 'from,' etc.
3. Show ownership or affiliation via terms like 'owned by,' 'affiliated with,' 'publisher of,' etc.
4. Offer identification or categorization like 'is,' 'are,' 'was,' etc.
Avoid stopwords.
---------------------
Example:
Question: When was the institute that owned The Collegian founded?
Text: The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas.
Triplets:
(The Collegian, is, bi-weekly official student publication)
(The Collegian, owned by, Houston Baptist University)
(Houston Baptist University, in, Houston)
(Houston, in, Texas)
---------------------
Text: {text}
Triplets:
""".strip()


def make_erx_prompt(model_type: str):
    prompt_str = LLAMA2_KG_TRIPLET_EXTRACT_TMPL if model_type == "llama2-sft" else DEFAULT_KG_TRIPLET_EXTRACT_TMPL
    return Prompt(
        prompt_str,
        prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
    )


def make_docs(example, only_supporting=False):
    ps = example["paragraphs"]
    for p in ps:
        if only_supporting and not p["is_supporting"]:
            continue
        idx = p["idx"]
        title = p["title"]
        body = p["paragraph_text"]
        is_supporting = p["is_supporting"]
        text = f"# {title}\n{body}"
        yield Document(text=text, metadata=dict(parent_id=example["id"], idx=idx, is_supporting=is_supporting))


def construct_knowledge_graph(
    example,
    *,
    max_triplets_per_chunk: int,
    include_embeddings: bool,
    model_type: str,
    trace_callback: Callable[[List[Span]], None],
    out_dir: Path,
):
    db = kuzu.Database(str(out_dir / "kuzu"))
    graph_store = KuzuGraphStore(db)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # documents to index into knowledge graph
    documents = list(make_docs(example, only_supporting=True))

    if model_type == "llama2-sft":
        from bellek.ml.kg.cons import parse_triplets

        def _parse_triplet_response(response: str, max_length: int = 128) -> list[tuple[str, str, str]]:
            return parse_triplets(response.strip())

        KnowledgeGraphIndex._parse_triplet_response = staticmethod(_parse_triplet_response)

    # extract triplets from documents
    return KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        storage_context=storage_context,
        service_context=make_service_context(model_type, trace_callback),
        include_embeddings=include_embeddings,
        kg_triple_extract_template=make_erx_prompt(model_type),
    )


def visualize_knowledge_graph(index, out: Path):
    g = index.get_networkx_graph()
    net = Network(notebook=False, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.save_graph(str(out))
    return out


def main(
    dataset_file: Path = typer.Option(...),
    model_type: str = typer.Option(...),
    out: Path = typer.Option(...),
):
    with open(dataset_file) as f:
        for line in f:
            example = json.loads(line)
            id = example["id"]

            example_out_dir = out / id
            shutil.rmtree(example_out_dir, ignore_errors=True)
            example_out_dir.mkdir(exist_ok=True, parents=True)

            try:
                err(f"Constructing the knowledge graph for the sample {id}")
                trace_callback = TraceRecorder((example_out_dir / "traces.jsonl").open("w"))
                index = construct_knowledge_graph(
                    example,
                    max_triplets_per_chunk=10,
                    include_embeddings=False,
                    model_type=model_type,
                    trace_callback=trace_callback,
                    out_dir=example_out_dir,
                )
                index.storage_context.persist(persist_dir=(example_out_dir / "index"))
                err(f"Visualizing the knowledge graph for the sample {id}")
                visualize_knowledge_graph(index, example_out_dir / "kuzu-network.html")
            except Exception as exc:
                err(f"Failed to construct the knowledge graph for the sample {id}.\n{exc}")

    (out / "timestamp.txt").write_text(str(datetime.now().isoformat(timespec="milliseconds")))


if __name__ == "__main__":
    typer.run(main)
