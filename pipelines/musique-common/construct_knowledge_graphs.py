import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import kuzu
import typer
from dotenv import load_dotenv
from llama_index import Document, KnowledgeGraphIndex, ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.storage_context import StorageContext
from more_itertools import partition
from pyvis.network import Network
from rich.console import Console
from tqdm import tqdm

from bellek.llama_index.graph_stores.kuzu import KuzuGraphStore
from bellek.llama_index.obs import make_phoenix_trace_callback_handler
from bellek.musique.constants import SKIPPED_RECORD_IDS
from bellek.utils import set_seed

err = Console(stderr=True).print

load_dotenv()

set_seed(89)


def make_trace_callback_handler(example_dir: Path):
    traces_filepath = example_dir / "traces.jsonl"
    traces_filepath.unlink(missing_ok=True)
    return make_phoenix_trace_callback_handler(traces_filepath)


def make_service_context(trace_callback_handler: BaseCallbackHandler):
    # model to generate embeddings for triplets
    embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # Setup LLM observability
    callback_manager = CallbackManager(handlers=[trace_callback_handler])

    return ServiceContext.from_defaults(
        embed_model=embed_model,
        transformations=[],
        callback_manager=callback_manager,
    )


def make_docs(example, only_supporting=False):
    non_supporting_ps, supporting_ps = partition(lambda p: p["is_supporting"], example["paragraphs"])
    ps = list(supporting_ps) if only_supporting else list(supporting_ps) + list(non_supporting_ps)
    for p in ps:
        idx = p["idx"]
        title = p["title"]
        body = p["paragraph_text"]
        is_supporting = p["is_supporting"]
        text = f"# {title}\n{body}"
        yield Document(
            text=text,
            metadata={"parent_id": example["id"], "idx": idx, "is_supporting": is_supporting},
            excluded_llm_metadata_keys=["parent_id", "idx", "is_supporting"],
        )


def make_kg_triplet_extract_fn_from_config(llm_config: dict[str, Any]):
    if llm_config["type"] == "offline":
        from bellek.jerx.offline.llm import make_kg_triplet_extract_fn

        return make_kg_triplet_extract_fn(llm_config["params"]["filepath"])
    elif llm_config["type"] == "online":
        from bellek.jerx.fewshot.llm import make_kg_triplet_extract_fn

        return make_kg_triplet_extract_fn(
            model=llm_config["model"],
            completion_params=llm_config.get("completion_params"),
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_config['type']}")


def visualize_knowledge_graph(index, out: Path):
    g = index.get_networkx_graph()
    net = Network(notebook=False, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.save_graph(str(out))
    return out


def construct_knowledge_graph(
    example,
    *,
    include_embeddings: bool,
    kg_triplet_extract_fn: Callable,
    service_context: ServiceContext,
    out_dir: Path,
):
    id = example["id"]

    db = kuzu.Database(str(out_dir / "kuzu"))
    graph_store = KuzuGraphStore(db)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Create documents to index into knowledge graph
    documents = list(make_docs(example, only_supporting=True))
    err(f"Created {len(documents)} documents for sample {id}")

    with open(out_dir / "documents.jsonl", "w") as f:
        for document in documents:
            triplets = kg_triplet_extract_fn(document.text)
            payload = document.dict()
            payload["triplets"] = triplets
            f.write(json.dumps(payload) + "\n")

    # Create knowledge graph index
    err(f"Creating the knowledge graph index for sample {id}")
    index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        include_embeddings=include_embeddings,
        kg_triplet_extract_fn=kg_triplet_extract_fn,
    )

    err(f"Persisting the knowledge graph index for sample {id}")
    index.storage_context.persist(persist_dir=(out_dir / "index"))

    err(f"Visualizing the knowledge graph for sample {id}")
    visualize_knowledge_graph(index, out_dir / "kuzu-network.html")


def process_example(
    example: dict,
    llm_config: dict,
    out: Path,
    ignore_errors: bool,
    resume: bool,
):
    example_id = example["id"]
    example_out_dir = out / example_id

    if example_id in SKIPPED_RECORD_IDS:
        return

    if resume and (example_out_dir / "kuzu-network.html").exists():
        err(f"Skipping the sample {example_id} as it already exists")
        return

    shutil.rmtree(example_out_dir, ignore_errors=True)
    example_out_dir.mkdir(exist_ok=True, parents=True)

    kg_triplet_extract_fn = make_kg_triplet_extract_fn_from_config(llm_config)

    try:
        err(f"Constructing the knowledge graph for the sample {example_id}")
        trace_callback_handler = make_trace_callback_handler(example_out_dir)
        service_context = make_service_context(trace_callback_handler)
        construct_knowledge_graph(
            example,
            include_embeddings=False,
            kg_triplet_extract_fn=kg_triplet_extract_fn,
            service_context=service_context,
            out_dir=example_out_dir,
        )
    except Exception as exc:
        err(f"Failed to construct the knowledge graph for sample {example_id}.\n{exc}")
        if not ignore_errors:
            raise exc


def main(
    dataset_file: Path = typer.Option(...),
    llm_config_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
    ignore_errors: bool = typer.Option(False),
    resume: bool = typer.Option(False),
):
    llm_config = json.loads(llm_config_file.read_text())

    with open(dataset_file) as f:
        examples = [json.loads(line) for line in f]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_example,
                example,
                llm_config=llm_config,
                out=out,
                ignore_errors=ignore_errors,
                resume=resume,
            )
            for example in examples
        ]
        for future in tqdm(as_completed(futures), total=len(examples), desc="Constructing knowledge graphs"):
            future.result()

    (out / "timestamp.txt").write_text(str(datetime.now().isoformat(timespec="milliseconds")))


if __name__ == "__main__":
    typer.run(main)
