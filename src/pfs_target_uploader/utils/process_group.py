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
    when the group's only remaining members are zombies not yet reaped --
    e.g. SIGTERM already killed them and SIGKILL follows before anything
    reaps them. Those are dead and hold nothing, so treat it like ESRCH.
    EPERM also means the signal was *not* delivered, so tolerating it cannot
    disturb anything we did not already own.
    """
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        logger.debug(
            f"killpg({pgid}, {sig}) refused with EPERM;"
            " treating the group as already dead"
        )


def terminate_process_group(proc, timeout=5.0):
    """Terminate ``proc`` and everything it forked, then reap it.

    Escalates to ``SIGKILL`` for anything still standing ``timeout`` seconds
    after the ``SIGTERM``. The group outlives its leader as long as it has
    members, so the second signal still reaches a grandchild that ignored the
    first one -- or one the child forked while it was being terminated.

    SIGKILL cannot be blocked, so every member is dead once this returns.
    Reaping them is init's job, and waiting for it here would hang wherever
    PID 1 does not reap (a container with no init and no supervisor loop).
    """
    if proc.pid is None:
        proc.join()
        return

    pgid = proc.pid

    if not proc.is_alive():
        # The leader is gone, but a crash or os._exit() skips multiprocessing's
        # finalizers, so its pool can still be running in the group it left
        # behind -- and with the leader dead, nothing else can name those
        # processes. proc.pid is a child's pid, never the pgid of our own
        # group, so this is safe to send blind; it reaches nothing if the
        # group is already empty.
        _signal_group(pgid, signal.SIGKILL)
        proc.join()
        return

    try:
        leads_own_group = os.getpgid(proc.pid) == pgid
    except ProcessLookupError:
        # Exited between the check above and here; same reasoning as above.
        _signal_group(pgid, signal.SIGKILL)
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

    try:
        _signal_group(pgid, signal.SIGTERM)
        proc.join(timeout)
    finally:
        # Escalate even if the wait was cut short -- a second Ctrl-C during
        # cleanup would otherwise leave the group half-signalled, orphaning
        # whatever ignored the SIGTERM.
        _signal_group(pgid, signal.SIGKILL)
        proc.join()
