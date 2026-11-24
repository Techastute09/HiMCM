from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "src"))
from overall_evaluation_option2 import run_full_evaluation


def main():
    run_full_evaluation(HERE)


if __name__ == "__main__":
    main()
