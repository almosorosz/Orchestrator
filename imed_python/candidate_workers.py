"""Process workers for the independent trials in a differential evolution search."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
from multiprocessing import get_context
import os


def max_cpu_workers():
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(count, 61) if os.name == "nt" else count


def _start_worker(evaluate, stop):
    global _evaluate
    _evaluate = partial(evaluate, cancelled=stop.is_set)


def _run_trial(candidate):
    return _evaluate(candidate)


@contextmanager
def candidate_map(workers, evaluate, collect, cancelled, *, completed=None):
    """Provide SciPy's map interface and keep progress reporting in the parent."""
    context = get_context("spawn")
    stop = context.Event()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_start_worker, initargs=(evaluate, stop))

    def check_stop():
        if cancelled is not None and cancelled():
            raise InterruptedError("Search stopped.")

    def map_trials(_scipy_objective, candidates):
        # The SciPy objective closes over parent state. Workers use their own evaluator.
        check_stop()
        pending = [(candidate, executor.submit(_run_trial, candidate)) for candidate in candidates]
        if completed is not None:
            # Notify on completion and collect in input order so draws and outputs stay aligned.
            unfinished = {trial for _, trial in pending}
            while unfinished:
                check_stop()
                finished, unfinished = wait(unfinished, timeout=0.1, return_when=FIRST_COMPLETED)
                for trial in finished:
                    completed(trial.result())
        scores = []
        for candidate, trial in pending:
            while not trial.done():
                check_stop()
                wait((trial,), timeout=0.1)
            check_stop()
            scores.append(collect(candidate, trial.result()))
        return scores

    try:
        yield map_trials
    finally:
        stop.set()
        executor.shutdown(cancel_futures=True)
