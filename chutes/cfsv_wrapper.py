import os
import sys
import stat
import subprocess
from pathlib import Path


def main():
    binary_path = Path(__file__).parent / "cfsv"
    if not binary_path.exists():
        print(f"Error: cfsv binary not found at {binary_path}", file=sys.stderr)
        sys.exit(1)
    try:
        os.chmod(binary_path, os.stat(binary_path).st_mode | stat.S_IEXEC)
    except OSError as e:
        print(f"Warning: Could not set executable permission on {binary_path}: {e}", file=sys.stderr)
    result = subprocess.run([str(binary_path)] + sys.argv[1:])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
