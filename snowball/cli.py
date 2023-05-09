import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from snowball.bootstrapping import Snowball


def create_args() -> ArgumentParser:  # pylint: disable=missing-function-docstring
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="file with bootstrapping configuration parameters", type=str, required=False)
    parser.add_argument(
        "--sentences",
        help="a text file with a sentence per line, and with at least two entities per sentence",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--positive_seeds",
        help="a text file with a seed per line, in the format, e.g.: 'Nokia;Espoo'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--negative_seeds",
        help="a text file with a seed per line, in the format, e.g.: 'Microsoft;San Francisco'",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--similarity",
        help="the minimum similarity between tuples and patterns to be considered a match",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--confidence",
        help="the minimum confidence score for a match to be considered a true positive",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--iterations",
        help="the minimum confidence score for a match to be considered a true positive",
        type=int,
        required=False,
        default=2,
    )

    return parser


def main() -> None:  # pylint: disable=missing-function-docstring
    parser = create_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    snowball = Snowball(
        args.config,
        args.positive_seeds,
        args.negative_seeds,
        args.sentences,
        args.similarity,
        args.confidence,
        args.iterations,
    )

    if args.sentences.endswith(".pkl"):
        print("Loading pre-processed sentences", args.sentences)
        snowball.init_bootstrap(args.sentences)
    else:
        snowball.generate_tuples(args.sentences)
        snowball.init_bootstrap(tuples=None)


if __name__ == "__main__":
    main()
