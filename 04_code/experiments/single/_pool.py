"""Worker count for the stage scripts: `--workers N` anywhere on the command line (removed from sys.argv on import),
default max(1, os.cpu_count() - 1). Seeds are per job, so results do not depend on N."""
import os
import sys

NWORK = max(1, (os.cpu_count() or 2) - 1)
if "--workers" in sys.argv:
    _i = sys.argv.index("--workers")
    NWORK = max(1, int(sys.argv[_i + 1]))
    del sys.argv[_i:_i + 2]
