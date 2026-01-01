"""Entry point for python -m nnviz."""

from .cli import parse_args
from .main import run_pipeline


def main():
    """Main entry point."""
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
