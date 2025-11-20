import sys
from pathlib import Path

# Add project root to Python path so `src` is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import load_sentences


def main():
    path = Path("data/intermediate/restaurants.jsonl")
    data = load_sentences(path)
    print("Loaded sentences:", len(data))
    print(data[0])


if __name__ == "__main__":
    main()
