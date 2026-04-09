"""Runtime helpers for bounded execution and progress-friendly diagnostics."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


class HardTimeoutError(TimeoutError):
    """Raised when a subprocess exceeds a hard runtime limit."""


@dataclass
class TimedResult:
    """Container for a result plus elapsed runtime."""

    value: Any
    elapsed_seconds: float


def _subprocess_entry(queue, func: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    try:
        result = func(**kwargs)
        queue.put({"ok": True, "result": result})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_with_hard_timeout(
    func: Callable[..., Any],
    *,
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: int,
    stage_name: str,
) -> TimedResult:
    """Run a top-level callable in a subprocess and terminate it on timeout."""
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_subprocess_entry, args=(queue, func, kwargs))

    start = perf_counter()
    process.start()
    process.join(timeout_seconds)
    elapsed = perf_counter() - start

    if process.is_alive():
        process.terminate()
        process.join(5)
        raise HardTimeoutError(f"{stage_name} exceeded {timeout_seconds}s.")

    if queue.empty():
        raise RuntimeError(f"{stage_name} exited without returning a result.")

    payload = queue.get()
    if payload.get("ok"):
        return TimedResult(value=payload.get("result"), elapsed_seconds=elapsed)

    error = payload.get("error", "unknown subprocess error")
    tb = payload.get("traceback", "")
    raise RuntimeError(f"{stage_name} failed: {error}\n{tb}".strip())
