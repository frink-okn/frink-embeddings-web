import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class OutputRecord:
    iris: list[str]
    label: str
    embedding_text: str
    # Total distinct IRIs that produced this text, before `iris` is capped at
    # `max_iris_per_record`. May exceed `len(iris)`.
    iri_count: int


@dataclass
class WorkerRecord:
    """One materialized root, before grouping by identical text.

    The per-root unit produced by `Textifier.materialize_one`; many of these
    are grouped into one `OutputRecord` by `digest` (= digest of the text).
    """

    iri: str
    label: str
    embedding_text: str
    digest: str


def write_json(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_json_record(record: OutputRecord, f, first: bool) -> bool:
    """Write one record into a streamed JSON array; returns the new `first`."""
    f.write("\n" if first else ",\n")
    text = json.dumps(asdict(record), indent=2, ensure_ascii=False)
    f.write("  ")
    f.write(text.replace("\n", "\n  "))
    return False


def write_jsonl(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            write_jsonl_record(record, f)


def write_jsonl_record(record: OutputRecord, f) -> None:
    json.dump(asdict(record), f, ensure_ascii=False)
    f.write("\n")


def write_text(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            write_text_record(record, f)


def write_text_record(record: OutputRecord, f) -> None:
    f.write(f"label: {record.label}\n")
    f.write("iris:\n")
    for iri in record.iris:
        f.write(f"- {iri}\n")
    f.write("\n")
    f.write(record.embedding_text)
    f.write("\n\n---\n\n")
