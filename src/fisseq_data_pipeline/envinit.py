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
    for filename in ("nextflow.config", "workflow.nf", "run.sh", "init.sh", "PIPELINE_README.md"):
        dest = target / filename
        dest.write_text((templates / filename).read_text())

    for script in ("run.sh", "init.sh"):
        (target / script).chmod(0o755)

    print(f"Initialised experiment directory: {target}")
    print(f"  {target}/input/             <- place raw .parquet batch files here")
    print(f"  {target}/init.sh            <- create venv and install the package")
    print(f"  {target}/run.sh             <- run the Nextflow pipeline")
    print(f"  {target}/nextflow.config    <- edit to configure your environment/profile")
    print(f"  {target}/PIPELINE_README.md <- pipeline usage documentation")
