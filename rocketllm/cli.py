"""Command line entry point.

``rocketllm profile`` exists so that a bug report can carry the machine it happened on. It prints
every probed measurement and every derived knob next to the formula that produced it, which is
usually enough to see why a number came out the way it did without anyone reading the source.
"""
import argparse
import json
import sys


def _add_profile_parser(subparsers):
    parser = subparsers.add_parser(
        "profile",
        help="probe this machine and print the hardware profile and every derived tuning knob")
    parser.add_argument("--weights-path", default=None,
                        help="probe storage on the filesystem holding these weights; without it "
                             "storage bandwidth cannot be measured and is reported as unavailable")
    parser.add_argument("--device", default=None,
                        help="override the auto-selected backend, e.g. cpu or cuda:1")
    parser.add_argument("--reprofile", action="store_true",
                        help="ignore any cached profile and measure again")
    parser.add_argument("--json", action="store_true",
                        help="emit the profile as JSON instead of the readable report")
    parser.add_argument("--storage-budget-seconds", type=float, default=3.0,
                        help="time budget for the storage sweep (default: 3.0)")
    parser.set_defaults(handler=_run_profile)
    return parser


def _run_profile(args):
    from .hw import HardwareProfile

    profile = HardwareProfile.load_or_probe(
        weights_path=args.weights_path,
        device=args.device,
        reprofile=args.reprofile,
        storage_budget_seconds=args.storage_budget_seconds)

    if args.json:
        json.dump(profile.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(profile.describe())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rocketllm", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command")
    _add_profile_parser(subparsers)

    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
