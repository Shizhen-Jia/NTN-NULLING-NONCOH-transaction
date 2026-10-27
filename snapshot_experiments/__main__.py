"""Optional command-line entry point; the root notebook is the primary interface."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from .config import SnapshotConfig, reference_music_config
from .workflow import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--source-run", default=SnapshotConfig.source_run)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    config = SnapshotConfig(source_run=args.source_run, num_scenes=args.scenes,
                            repeats=args.repeats, workers=args.workers)
    directory = execute(config, reference_music_config(), resume_directory=args.resume,
                        smoke=not args.skip_smoke)
    print(f"Completed: {directory}", flush=True)


if __name__ == "__main__":
    main()

