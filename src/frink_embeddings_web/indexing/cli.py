from pathlib import Path
from typing import Annotated

import typer

from .index import (
    load_graph,
    materialize_records,
    write_json,
    write_text,
)
from .models import MaterializationConfiguration

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.callback()
def main():
    pass


@app.command()
def textify(
    hdt_file: Annotated[
        Path,
        typer.Argument(help="Input HDT graph file."),
    ],
    config_toml: Annotated[
        Path,
        typer.Argument(help="Indexing TOML config."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path."),
    ],
    text: Annotated[
        bool,
        typer.Option(
            "--text",
            help="Write debug text output instead of JSON.",
        ),
    ] = False,
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="Name of one target from the config to materialize.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of root nodes to process per target.",
        ),
    ] = None,
):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)
    records = materialize_records(
        graph,
        config,
        target=target,
        limit=limit,
    )

    if text:
        write_text(records, output)
    else:
        write_json(records, output)

    typer.echo(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    app()
