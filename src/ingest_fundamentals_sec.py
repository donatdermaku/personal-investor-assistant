import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.ingest.fundamentals_sec import main


if __name__ == "__main__":
    main()
