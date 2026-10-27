"""Run the frozen-channel activity study, then its analysis and synthetic controls."""
import argparse
import json
from pathlib import Path
from .config import ActivityConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'analyze', 'synthetic', 'plot', 'finish'])
    parser.add_argument('--directory')
    parser.add_argument('--config', help='Optional JSON file with ActivityConfig fields')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'run':
        from .runner import run
        values=json.loads(Path(args.config).read_text()) if args.config else {}
        values['workers']=args.workers
        print(run(ActivityConfig(**values), resume=args.directory))
    elif args.action == 'analyze':
        from .analysis import analyze
        analyze(args.directory)
    elif args.action == 'synthetic':
        from .synthetic import run_synthetic
        run_synthetic(args.directory)
    elif args.action == 'plot':
        from .plotting import plot
        plot(args.directory)
    else:
        from .finalize import finish
        finish(args.directory)


if __name__ == '__main__':
    main()
