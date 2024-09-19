from pathlib import Path

base_path: Path = Path(__file__).resolve

CONTROLLERS_PATH: Path = Path(__file__).resolve().parent.parent.parent.joinpath("controllers")
