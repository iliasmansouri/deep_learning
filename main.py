from pathlib import Path
from train import TrainingLauncher
import typer


def main(model_name: str, path_to_data: Path):
    launcher = TrainingLauncher(path_to_data, model_name)
    launcher.train()


if __name__ == "__main__":
    typer.run(main)
