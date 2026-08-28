"""Run a subprocess in its own process group so it can be cleaned up whole.

PPP is run as an ``mp.Process`` that forks further children of its own (the
KDE worker pool in :mod:`~pfs_target_uploader.utils.ppp`). Terminating that
tree by walking it with psutil is racy: the pool's maintenance thread re-forks
a worker as soon as one dies, so a worker terminated from a snapshot of the
tree is replaced while the parent is still alive, and the replacement is
orphaned the moment the parent goes away.

Giving the subprocess its own process group removes the race instead of
narrowing it -- ``killpg`` reaches every member, including one forked a
microsecond earlier.
"""

import multiprocessing as mp
import os
import signal
import time

from loguru import logger


def _process_group_entrypoint(target, args):
    # Runs in the child. setpgrp() makes it the leader of a new group whose id
    # is its own pid, so the caller can address the whole tree by proc.pid.
    os.setpgrp()
    target(*args)


def start_in_process_group(target, args=(), name=None):
    """Start ``target(*args)`` in a subprocess that leads its own process group."""
    proc = mp.Process(target=_process_group_entrypoint, name=name, args=(target, args))
    proc.start()
    return proc


def _signal_group(pgid, sig):
    """Signal every member of ``pgid``, tolerating a group that is already gone.

    Every member is our own descendant running as us, so a permission error
    cannot mean someone else's process: macOS raises EPERM rather than ESRCH
    for a real signal too (not just signal 0, see _wait_for_group_to_clear)
    when the group's only remaining members are zombies not yet reaped --
    e.g. SIGTERM already killed them and SIGKILL follows before anything
    reaps them. Those are dead and hold nothing, so treat it like ESRCH.
    """
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_for_group_to_clear(pgid, timeout, interval=0.02):
    """Block until no member of ``pgid`` is left, or ``timeout`` runs out.

    Signal 0 reports whether the group still has a member. A killed process
    leaves the group as soon as it is reaped, and an orphan is reaped by init
    as soon as its parent is gone, so this normally returns within a tick.
    See _signal_group for why a PermissionError here also means clear.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def terminate_process_group(proc, timeout=5.0):
    """Terminate ``proc`` and everything it forked, then reap it.

    Escalates to ``SIGKILL`` for anything still standing ``timeout`` seconds
    after the ``SIGTERM``. The group outlives its leader as long as it has
    members, so the second signal still reaches a grandchild that ignored the
    first one -- or one the child forked while it was being terminated. Waits
    for the group to empty before returning, so callers can rely on nothing
    being left behind.
    """
    if proc.pid is None or not proc.is_alive():
        proc.join()
        return

    pgid = proc.pid

    try:
        leads_own_group = os.getpgid(proc.pid) == pgid
    except ProcessLookupError:
        proc.join()
        return

    if not leads_own_group:
        # The child has not reached setpgrp() yet, so the only group it
        # belongs to is ours -- the one running the Panel server or the CLI.
        # Signalling that would take the caller down with it.
        logger.warning(
            f"{proc.name} (pid {proc.pid}) leads no process group of its own;"
            " terminating the process alone"
        )
        proc.terminate()
        proc.join(timeout)
        proc.kill()
        proc.join()
        return

    _signal_group(pgid, signal.SIGTERM)
    proc.join(timeout)
    _signal_group(pgid, signal.SIGKILL)
    proc.join()

    if not _wait_for_group_to_clear(pgid, timeout):
        logger.warning(
            f"process group {pgid} ({proc.name}) still has members"
            f" {timeout:.0f} s after SIGKILL"
        )
