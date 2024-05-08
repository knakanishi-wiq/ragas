from __future__ import annotations

import asyncio
import logging
import sys
import threading
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.exceptions import MaxRetriesExceeded
from ragas.run_config import RunConfig

from loguru import logger


def runner_exception_hook(args: threading.ExceptHookArgs):
    print(args)
    raise args.exc_type


# set a custom exception hook
# threading.excepthook = runner_exception_hook


def as_completed(loop, coros, max_workers):
    loop_arg_dict = {"loop": loop} if sys.version_info[:2] < (3, 10) else {}
    if max_workers == -1:
        return asyncio.as_completed(coros, **loop_arg_dict)

    # loop argument is removed since Python 3.10
    semaphore = asyncio.Semaphore(max_workers, **loop_arg_dict)

    async def sema_coro(coro):
        async with semaphore:
            return await coro

    sema_coros = [sema_coro(c) for c in coros]
    return asyncio.as_completed(sema_coros, **loop_arg_dict)


class Runner(threading.Thread):
    def __init__(
        self,
        jobs: t.List[t.Tuple[t.Coroutine, str]],
        desc: str,
        keep_progress_bar: bool = True,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        super().__init__()
        self.jobs = jobs
        self.desc = desc
        self.keep_progress_bar = keep_progress_bar
        self.raise_exceptions = raise_exceptions
        self.run_config = run_config or RunConfig()

        # create task
        self.loop = asyncio.new_event_loop()
        self.futures = as_completed(
            loop=self.loop,
            coros=[coro for coro, _ in self.jobs],
            max_workers=self.run_config.max_workers,
        )

    async def _aresults(self) -> t.List[t.Any]:
        results = []
        for future in tqdm(
            self.futures,
            desc=self.desc,
            total=len(self.jobs),
            # whether you want to keep the progress bar after completion
            leave=self.keep_progress_bar,
        ):
            r = (-1, np.nan)
            try:
                r = await future
            except MaxRetriesExceeded as e:
                logger.warning(f"max retries exceeded for {e.evolution}")
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    logger.error(
                        "Runner in Executor raised an exception", exc_info=True
                    )
            results.append(r)

        return results

    def run(self):
        results = []
        try:
            results = self.loop.run_until_complete(self._aresults())
        finally:
            self.results = results
            self.loop.stop()


@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = field(default_factory=RunConfig, repr=False)

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        return wrapped_callable_async

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index(*args, **kwargs), name))

    def results(self) -> t.List[t.Any]:
        executor_job = Runner(
            jobs=self.jobs,
            desc=self.desc,
            keep_progress_bar=self.keep_progress_bar,
            raise_exceptions=self.raise_exceptions,
            run_config=self.run_config,
        )
        executor_job.start()
        try:
            executor_job.join()
        finally:
            ...

        if executor_job.results is None:
            if self.raise_exceptions:
                raise RuntimeError(
                    "Executor failed to complete. Please check logs above for full info."
                )
            else:
                logger.error("Executor failed to complete. Please check logs above.")
                return []
        sorted_results = sorted(executor_job.results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]


@dataclass
class MainThreadRunner:
    """
    A class to manage and run jobs on the main thread only.

    Attributes:
        jobs (List[Tuple[Coroutine, str]]): A list of jobs to be run. Each job is a tuple containing a coroutine and a string.
        desc (str): A description of the runner.
        keep_progress_bar (bool): A flag indicating whether to keep the progress bar after completion. Default is True.
        raise_exceptions (bool): A flag indicating whether to raise exceptions. Default is True.
        run_config (Optional[RunConfig]): An optional configuration for the run.

    Methods:
        _main_thread_as_completed(coros: List[Coroutine], max_workers: int) -> asyncio.as_completed: An internal method that returns an iterator that yields futures as they complete.
        _aresults() -> List[Any]: An asynchronous method that runs all jobs and returns a list of results.
        run() -> List[Any]: A method that starts the event loop and runs all jobs until completion.
    """

    jobs: t.List[t.Tuple[t.Coroutine, str]]
    desc: str
    keep_progress_bar: bool = True
    raise_exceptions: bool = True
    run_config: t.Optional[RunConfig] = None

    def _main_thread_as_completed(self, coros: t.List[t.Coroutine], max_workers: int) -> asyncio.as_completed:
        """
        This function returns an iterator that yields futures as they complete.
        If max_workers is -1, it returns all the coroutines as they complete.
        If max_workers is not -1, it uses a semaphore to limit the number of coroutines
        that can run at the same time to max_workers.

        Args:
            coros (List[Coroutine]): A list of coroutine objects that you want to run.
            max_workers (int): The maximum number of coroutines that can run at the same time.
                               If it's -1, all coroutines are run at the same time.

        Returns:
            asyncio.as_completed: An iterator that yields futures as they complete.
        """
        if max_workers == -1:
            return asyncio.as_completed(coros)

        semaphore = asyncio.Semaphore(max_workers)

        async def sema_coro(coro: t.Coroutine) -> t.Any:
            """
            This is an internal coroutine function that runs the given coroutine
            while acquiring the semaphore.

            Args:
                coro (Coroutine): The coroutine that you want to run.

            Returns:
                Any: The result of the coroutine execution.
            """
            async with semaphore:
                return await coro

        sema_coros = [sema_coro(c) for c in coros]
        return asyncio.as_completed(sema_coros)

    async def _aresults(self) -> t.List[t.Any]:
        """
        Asynchronously runs all jobs and collects the results.

        This method creates futures for all jobs and runs them using an event loop.
        It collects the results of each job as they complete and returns a list of results.

        Returns:
            List[Any]: A list of results from all jobs.
        """
        results = []
        futures = self._main_thread_as_completed(
            [coro for coro, _ in self.jobs],
            max_workers=self.run_config.max_workers if self.run_config else -1,
        )
        for future in tqdm(
            futures,
            desc=self.desc,
            total=len(self.jobs),
            leave=self.keep_progress_bar,
        ):
            r = (-1, np.nan)
            try:
                r = await future
            except MaxRetriesExceeded as e:
                logger.warning(f"max retries exceeded for {e.evolution}")
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    logger.error("Runner encountered an exception", exc_info=True)
            results.append(r)

        return results

    def run(self) -> t.List[t.Any]:
        """
        Gets the event loop and runs all jobs until completion.

        This method gets the main event loop, runs the _aresults method until completion,
        and returns the results.

        Returns:
            List[Any]: A list of results from all jobs.
        """
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._aresults())
        return results


@dataclass
class MainThreadExecutor:
    """
    A class to create and execute async jobs on the main thread.

    Attributes:
        desc (str): A description of the executor. Default is "Evaluating".
        keep_progress_bar (bool): A flag indicating whether to keep the progress bar after completion. Default is True.
        jobs (List[Any]): A list of jobs to be executed. Default is an empty list.
        raise_exceptions (bool): A flag indicating whether to raise exceptions. Default is False.
        run_config (Optional[RunConfig]): An optional configuration for the run. Default is a new RunConfig instance.

    """

    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = field(default_factory=RunConfig, repr=False)

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        """
        Wraps a callable with an index.

        This method takes a callable and a counter as input, and returns a new callable that, when called,
        returns the counter and the result of the original callable.

        Args:
            callable (Callable): The original callable to be wrapped.
            counter (int): The counter to be returned along with the result of the callable.

        Returns:
            Callable: The wrapped callable.
        """

        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        return wrapped_callable_async

    def submit(self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs):
        """
        Submits a job to be executed.

        This method takes a callable and its arguments, wraps the callable with an index, and adds the job to the list of jobs.

        Args:
            callable (Callable): The callable to be executed.
            *args: The arguments to be passed to the callable.
            name (Optional[str]): An optional name for the job.
            **kwargs: The keyword arguments to be passed to the callable.
        """
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index(*args, **kwargs), name))

    def results(self) -> t.List[t.Any]:
        """
        Executes all jobs and returns the results.

        This method creates a MainThreadRunner with the current jobs and configuration, runs all jobs, sorts the results by index,
        and returns the results.

        Returns:
            List[Any]: A list of results from all jobs.
        """
        runner = MainThreadRunner(
            jobs=self.jobs,
            desc=self.desc,
            keep_progress_bar=self.keep_progress_bar,
            raise_exceptions=self.raise_exceptions,
            run_config=self.run_config,
        )
        results = runner.run()
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
