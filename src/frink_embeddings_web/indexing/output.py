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


def write_json(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in records],
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            write_jsonl_record(record, f)


def write_jsonl_record(record: OutputRecord, f) -> None:
    json.dump(asdict(record), f, ensure_ascii=False)
    f.write("\n")


def write_text(records: Iterable[OutputRecord], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"label: {r.label}\n")
            f.write("iris:\n")
            for iri in r.iris:
                f.write(f"- {iri}\n")
            f.write("\n")
            f.write(r.embedding_text)
            f.write("\n\n---\n\n")
