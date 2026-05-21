import argparse
import importlib.resources
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fisseq-env-init",
        description="Initialise a FISSEQ experiment directory.",
    )
    parser.add_argument(
        "target_dir",
        help="Directory to initialise (created if it does not exist).",
    )
    args = parser.parse_args()

    target = Path(args.target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "input").mkdir(exist_ok=True)

    templates = importlib.resources.files("fisseq_data_pipeline.workflows")
    for filename in ("nextflow.config", "fisseq_pipeline.nf", "PIPELINE_README.md"):
        (target / filename).write_text((templates / filename).read_text())

    print(f"Initialised experiment directory: {target}")
    print(f"  {target}/input/               <- place raw .parquet batch files here")
    print(f"  {target}/fisseq_pipeline.nf   <- Nextflow pipeline (do not edit)")
    print(
        f"  {target}/nextflow.config      <- edit to configure your environment/profile"
    )
    print(f"  {target}/PIPELINE_README.md   <- pipeline usage documentation")
