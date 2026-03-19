import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from frink_embeddings_web.eval import Evaluation


def render_report(input_json: Path):
    env = Environment(
        loader=PackageLoader("frink_embeddings_web"),
        autoescape=select_autoescape(),
    )

    template = env.get_template("eval_report.md")

    with input_json.open() as fp:
        data = json.load(fp)
        evaluation = Evaluation.model_validate(data)

    print(template.render(**evaluation.model_dump()))


if __name__ == "__main__":
    render_report(Path("output.json"))
