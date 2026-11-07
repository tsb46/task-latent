"""
To-be-implemented main script for fitting a regression on the IBC dataset and saving R2 and brain maps.
"""

import argparse


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError("The main function is not yet implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a regression on the IBC dataset and save R2 and brain maps."
    )
    parser.add_argument(
        "-s", "--subject", type=str, required=True, help="Subject ID (e.g., '04')"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="ibc", help="Dataset name (default: ibc)"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, help="Verbosity level (default: 1)"
    )

    main(parser.parse_args())
