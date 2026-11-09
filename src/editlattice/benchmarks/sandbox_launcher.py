"""Run an untrusted Python program under resource limits.

Usage: python -I sandbox_launcher.py <cpu_seconds> <max_file_bytes> <program.py>

The limits are set inside the child process itself, because ``preexec_fn`` is not safe
in a multithreaded parent (our runner uses worker threads).
"""

import resource
import runpy
import sys


def apply_limits(cpu_seconds: int, max_file_bytes: int) -> None:
    """CPU time, written-file size and process count limits."""
    # Soft limit sends SIGXCPU at cpu_seconds; the hard limit (+1 s) kills with SIGKILL.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass


def main() -> None:
    """Apply limits, then execute the program as ``__main__``."""
    cpu_seconds = int(sys.argv[1])
    max_file_bytes = int(sys.argv[2])
    program = sys.argv[3]
    apply_limits(cpu_seconds, max_file_bytes)
    sys.argv = [program]
    runpy.run_path(program, run_name="__main__")


if __name__ == "__main__":
    main()
