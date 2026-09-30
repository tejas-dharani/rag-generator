"""Evaluate retrieval and answer quality on the sample document sets.

Usage:
    uv run python -m eval.run_eval              # full eval (needs an LLM, e.g. OPENAI_API_KEY)
    uv run python -m eval.run_eval --no-llm     # retrieval metrics only, no API key needed

Builds fresh collections in a temporary directory, so it never touches your app data.
"""

import argparse
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import yaml

from app.config import get_settings
from app.generation.prompts import NOT_FOUND
from app.service import RAGService

ROOT = Path(__file__).resolve().parent.parent


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "")).lower()


def build(service: RAGService, spec: dict) -> str:
    collection = service.create_collection(spec["name"])
    for path in sorted((ROOT / spec["folder"]).iterdir()):
        doc, _ = service.register_upload(collection.id, path.name, path.read_bytes())
        service.ingest_document(doc.id)
        doc = service.metadata.get_document(doc.id)
        if doc.status != "ready":
            sys.exit(f"Failed to ingest {path.name}: {doc.error}")
    return collection.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-llm", action="store_true", help="only measure retrieval")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.yaml"))
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.questions).read_text())
    base = get_settings()
    rows = []
    refusals = []

    with tempfile.TemporaryDirectory() as tmp:
        settings = base.model_copy(update={"data_dir": Path(tmp), "models_cache_dir": base.models_dir})
        service = RAGService(settings)
        mode = "retrieval only" if args.no_llm else f"{settings.llm_provider} · {settings.llm_model}"
        print(f"Evaluating with {settings.embedding_model} + reranker, LLM: {mode}\n")

        for collection_spec in spec["collections"]:
            cid = build(service, collection_spec)
            print(f"## {collection_spec['name']}")

            for q in collection_spec["questions"]:
                start = time.time()
                if args.no_llm:
                    chunks = service.retrieve(cid, q["question"])
                    files = [c.filename for c in chunks]
                    answer, cited_files = "", []
                else:
                    result = service.answer(cid, q["question"])
                    files = [s["filename"] for s in result["sources"]]
                    answer = result["answer"]
                    cited_files = [result["sources"][n - 1]["filename"] for n in result["cited"]]
                row = {
                    "question": q["question"],
                    "hit": q["expect_file"] in files,
                    "correct": all(normalize(p) in normalize(answer) for p in q["expect_all"]),
                    "cited_ok": q["expect_file"] in cited_files,
                    "seconds": time.time() - start,
                }
                rows.append(row)
                marks = f"retrieved {'✓' if row['hit'] else '✗'}"
                if not args.no_llm:
                    marks += f"  answer {'✓' if row['correct'] else '✗'}  cited {'✓' if row['cited_ok'] else '✗'}"
                print(f"  {marks}  {q['question']}")
                if not args.no_llm and not row["correct"]:
                    print(f"      got: {answer[:200]!r}")

            # Refusal is decided by the LLM, so it can only be measured with one.
            for question in [] if args.no_llm else collection_spec.get("out_of_scope", []):
                refused = NOT_FOUND.lower()[:25] in service.answer(cid, question)["answer"].lower()
                refusals.append(refused)
                print(f"  refused {'✓' if refused else '✗'}  (out of scope) {question}")
            print()

    def pct(values: list[bool]) -> str:
        return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)"

    print("## Summary")
    print(f"  Retrieval hit rate @k : {pct([r['hit'] for r in rows])}")
    if not args.no_llm:
        print(f"  Answer correctness    : {pct([r['correct'] for r in rows])}")
        print(f"  Correct citation      : {pct([r['cited_ok'] for r in rows])}")
        print(f"  Out-of-scope refusal  : {pct(refusals)}")
    print(f"  Median latency        : {statistics.median(r['seconds'] for r in rows):.1f}s per question")


if __name__ == "__main__":
    main()
