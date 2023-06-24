#!/usr/bin/env python

from __future__ import annotations

import argparse
import multiprocessing
import re
import subprocess
import contextlib
import time
from types import TracebackType
from typing import Type

import matplotlib.pyplot as plt
import numpy as np


class timeit(contextlib.AbstractContextManager):
    _elapsed_time: int
    _elapsed: bool

    def __enter__(self):
        self._elapsed_time = time.time_ns()
        self._elapsed = False

        return self

    def __exit__(self, __exc_type: Type[BaseException] | None,
                 __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self._elapsed_time = time.time_ns() - self._elapsed_time
        self._elapsed = True

        return

    @property
    def elapsed_time(self) -> int:
        if not self._elapsed:
            raise RuntimeError
        return self._elapsed_time


def main():
    parser = argparse.ArgumentParser(
        prog='run_benchmarks')

    parser.add_argument('-mint', '--min-threads', type=int, default=1)
    parser.add_argument('-maxt', '--max-threads', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-a', '--attempts', type=int, default=10)
    parser.add_argument('-s', '--step', type=int, default=1)
    parser.add_argument('-v', '--view', type=int, default=1)
    parser.add_argument('-e', '--efficient', type=bool, default=False)

    args = parser.parse_args()

    subprocess.run(['make'])
    times = []
    n_threads_list = list(range(args.min_threads, args.max_threads, args.step))
    pattern = re.compile(r'(\[\d+\.\d+\])')
    for i in n_threads_list:
        min_elapsed_time = float('inf')
        for _ in range(args.attempts):
            output = subprocess.check_output(
                ['./mandelbrot', '-t', str(i), '-v', str(args.view), '-e', str(int(args.efficient))]).decode('utf-8')
            elapsed_time = float(pattern.findall(output.replace('\n', ' '))[1][1:-1])
            min_elapsed_time = min(min_elapsed_time, elapsed_time)

        times.append(min_elapsed_time)

    times = np.array(times)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax in axes:
        ax.set_xlabel('Number of threads')

    axes[0].plot(n_threads_list, times)
    axes[0].set_ylabel('Elapsed time, ms')

    speedup = times[0] / times
    axes[1].plot(n_threads_list, speedup)
    axes[1].set_ylabel('Speedup')

    axes[2].plot(n_threads_list, (speedup / np.array(n_threads_list)) * 100)
    axes[2].set_ylabel('Efficiency, %')

    fig.savefig(f'view_{args.view}_bench_{"efficient" if args.efficient else "base"}.png')


if __name__ == '__main__':
    main()
