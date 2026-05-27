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
from .sample import (
    sample_targets,
    sample_types,
    write_sample_targets_json,
    write_sample_targets_text,
    write_sample_types_json,
    write_sample_types_text,
)

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


@app.command("sample-types")
def sample_types_cmd(
    hdt_file: Annotated[
        Path,
        typer.Argument(help="Input HDT graph file."),
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
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of subject IRIs to sample per type.",
        ),
    ] = 5,
    values_limit: Annotated[
        int,
        typer.Option(
            "--values-limit",
            min=1,
            help="Maximum literal examples to include per predicate.",
        ),
    ] = 3,
):
    graph = load_graph(hdt_file)
    records = sample_types(
        graph,
        limit=limit,
        values_limit=values_limit,
    )

    if text:
        write_sample_types_text(records, output)
    else:
        write_sample_types_json(records, output)

    typer.echo(f"Wrote {len(records)} type samples to {output}")


@app.command("sample-targets")
def sample_targets_cmd(
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
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of root nodes to process per target.",
        ),
    ] = 5,
):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)
    records = sample_targets(graph, config, limit=limit)

    if text:
        write_sample_targets_text(records, output)
    else:
        write_sample_targets_json(records, output)

    typer.echo(f"Wrote {len(records)} target samples to {output}")


if __name__ == "__main__":
    app()
