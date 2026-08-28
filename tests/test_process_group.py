import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from pfs_target_uploader.utils.process_group import (
    _signal_group,
    start_in_process_group,
    terminate_process_group,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="process groups are POSIX-only"
)

# Every process these tests start arms an alarm before it blocks, so a test
# that fails (or a cleanup that regresses) cannot leave a permanent orphan
# behind -- which is the very thing being fixed here.
SELF_DESTRUCT_SECONDS = 60

# Stand-in for the kde_pool's worker count.
POOL_SIZE = 4


def _sleep_until_self_destruct():
    """Block until SIGALRM kills us. Never returns."""
    signal.alarm(SELF_DESTRUCT_SECONDS)
    time.sleep(SELF_DESTRUCT_SECONDS * 2)
    os._exit(0)


def _plain_sleeper_child():
    """Body of a forked grandchild: drop inherited handlers, then block."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _sleep_until_self_destruct()


def _record_pid(pid_file, pid):
    with open(pid_file, "a") as f:
        f.write(f"{pid}\n")
        f.flush()


def _fork_a_replacement_on_sigterm(pid_file):
    """Fork a pool of children, and one more when cleanup comes knocking.

    Stands in for PPP's kde_pool: mp.Pool's maintenance thread re-forks a
    worker the moment one exits, so a cleanup that terminates the workers
    makes the parent create a process that no earlier snapshot of the tree can
    name -- and that outlives the parent. Every pid is recorded as it is
    forked, replacement included.
    """

    def _replace(signum, frame):
        pid = os.fork()
        if pid == 0:
            _plain_sleeper_child()
        _record_pid(pid_file, pid)
        os._exit(0)

    signal.signal(signal.SIGTERM, _replace)

    for _ in range(POOL_SIZE):
        pid = os.fork()
        if pid == 0:
            _plain_sleeper_child()
        _record_pid(pid_file, pid)

    _sleep_until_self_destruct()


def _fork_a_grandchild_then_die_abruptly(pid_file):
    """Fork one child, then exit without running any cleanup.

    A crash or os._exit() skips multiprocessing's atexit finalizers, so the
    KDE pool would outlive PPPrunStart the same way -- leaving a leaderless
    group whose members nothing walks any more.
    """
    pid = os.fork()
    if pid == 0:
        _plain_sleeper_child()
    _record_pid(pid_file, pid)
    os._exit(0)


class _InterruptOnFirstJoin:
    """A real process whose first join() raises, standing in for a second Ctrl-C."""

    def __init__(self, proc):
        self._proc = proc
        self._joins = 0

    def __getattr__(self, name):
        return getattr(self._proc, name)

    def join(self, timeout=None):
        self._joins += 1
        if self._joins == 1:
            raise KeyboardInterrupt
        return self._proc.join(timeout)


def _ignore_sigterm_until_self_destruct(ready_file):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(ready_file, "w") as f:
        f.write("ready")
        f.flush()
    _sleep_until_self_destruct()


def _exit_immediately():
    return


def _put_then_sleep(queue, value):
    queue.put(value)
    _sleep_until_self_destruct()


def _read_pids(pid_file):
    with open(pid_file) as f:
        return [int(line) for line in f.read().split()]


def _is_running(pid):
    """True if pid is a live process, i.e. not gone and not a reaped-pending zombie."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_start_in_process_group_makes_the_child_a_group_leader():
    proc = start_in_process_group(
        _sleep_until_self_destruct, args=(), name="test-leader"
    )
    try:
        assert _wait_until(
            lambda: os.getpgid(proc.pid) == proc.pid
        ), "child never became the leader of its own process group"
        assert os.getpgid(proc.pid) != os.getpgid(
            0
        ), "child is still in the parent's process group"
    finally:
        terminate_process_group(proc)


def test_terminate_process_group_kills_descendants_forked_after_a_snapshot(tmp_path):
    pid_file = tmp_path / "grandchildren.txt"
    pid_file.touch()

    proc = start_in_process_group(
        _fork_a_replacement_on_sigterm, args=(str(pid_file),), name="test-forker"
    )
    assert _wait_until(
        lambda: len(_read_pids(pid_file)) >= POOL_SIZE
    ), "child never forked its pool of grandchildren"

    terminate_process_group(proc)

    # Read the pids only now: the replacement the child forks while it is
    # being terminated is the one a snapshot of the tree can never name.
    assert not proc.is_alive()
    pids = _read_pids(pid_file)
    assert len(pids) == POOL_SIZE + 1, "the child never forked its replacement"
    # A process on its way out can still show up briefly in the process table,
    # so require survival to persist: a leaked one sleeps for a minute.
    assert _wait_until(
        lambda: not [pid for pid in pids if _is_running(pid)], timeout=5.0
    ), f"grandchildren survived the group kill: {[p for p in pids if _is_running(p)]}"


def test_terminate_process_group_escalates_to_sigkill(tmp_path):
    ready_file = tmp_path / "ready"
    proc = start_in_process_group(
        _ignore_sigterm_until_self_destruct,
        args=(str(ready_file),),
        name="test-stubborn",
    )
    assert _wait_until(ready_file.exists), "child never installed its SIGTERM handler"

    started = time.monotonic()
    terminate_process_group(proc, timeout=1.0)
    elapsed = time.monotonic() - started

    assert not proc.is_alive()
    assert (
        elapsed < 10.0
    ), f"waited {elapsed:.1f} s: a child that ignores SIGTERM was not killed"


def test_terminate_process_group_kills_what_a_dead_leader_left_behind(tmp_path):
    # The leader can die on its own between the caller's is_alive() check and
    # this call, or crash outright -- either way its group members are still
    # running and nothing else can name them.
    pid_file = tmp_path / "grandchild.txt"
    pid_file.touch()
    proc = start_in_process_group(
        _fork_a_grandchild_then_die_abruptly,
        args=(str(pid_file),),
        name="test-crasher",
    )
    assert _wait_until(lambda: len(_read_pids(pid_file)) >= 1), "no grandchild forked"
    assert _wait_until(lambda: not proc.is_alive()), "leader never exited"
    orphan = _read_pids(pid_file)[0]
    assert _is_running(orphan), "grandchild should outlive its leader here"

    terminate_process_group(proc)

    assert _wait_until(
        lambda: not _is_running(orphan), timeout=5.0
    ), "a dead leader's group members were left running"


def test_terminate_process_group_escalates_even_when_the_wait_is_interrupted(tmp_path):
    # A second Ctrl-C lands while terminate_process_group() waits out the
    # SIGTERM. The escalation still has to happen, or a child that ignores
    # SIGTERM is orphaned -- exactly what this function exists to prevent.
    ready_file = tmp_path / "ready"
    proc = start_in_process_group(
        _ignore_sigterm_until_self_destruct,
        args=(str(ready_file),),
        name="test-interrupted",
    )
    assert _wait_until(ready_file.exists), "child never installed its SIGTERM handler"

    with pytest.raises(KeyboardInterrupt):
        terminate_process_group(_InterruptOnFirstJoin(proc), timeout=1.0)

    assert _wait_until(
        lambda: not _is_running(proc.pid), timeout=5.0
    ), "SIGKILL escalation was skipped when the wait was interrupted"


def test_terminate_process_group_falls_back_for_a_child_that_leads_no_group():
    # A child is briefly still in our own group between start() and the
    # setpgrp() call in the entrypoint. Cleanup during that window must fall
    # back to terminating the process itself, and must never signal the group
    # it shares with us -- that group holds the Panel server or the CLI.
    bystander = mp.Process(target=_sleep_until_self_destruct, name="test-bystander")
    bystander.start()
    victim = mp.Process(target=_sleep_until_self_destruct, name="test-victim")
    victim.start()

    try:
        assert os.getpgid(victim.pid) == os.getpgid(0), "victim is not in our group"

        started = time.monotonic()
        terminate_process_group(victim, timeout=1.0)
        elapsed = time.monotonic() - started

        assert not victim.is_alive()
        assert (
            elapsed < 10.0
        ), f"waited {elapsed:.1f} s: cleanup did not terminate a non-leader child"
        assert bystander.is_alive(), "cleanup signalled our own process group"
    finally:
        bystander.kill()
        bystander.join()


def test_signal_group_tolerates_a_zombie_only_group():
    # Reproduces a crash seen in manual CLI testing: by the time SIGKILL is
    # sent, SIGTERM has already killed every real member and none has been
    # reaped yet. On macOS, killpg on a group whose only members are zombies
    # raises EPERM rather than ESRCH, so _signal_group has to tolerate more
    # than the "group is entirely gone" case.
    #
    # start_new_session gives the child its own group without forking this
    # multi-threaded pytest process; never wait()ing on it leaves the zombie.
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)

    try:
        assert _wait_until(
            lambda: psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE
        ), "child never became a zombie"

        _signal_group(child.pid, signal.SIGKILL)  # must not raise
    finally:
        child.wait()


def test_terminate_process_group_accepts_an_already_exited_process():
    proc = start_in_process_group(_exit_immediately, args=(), name="test-quick")
    proc.join(timeout=30)
    assert not proc.is_alive()

    terminate_process_group(proc)

    assert proc.exitcode == 0


def test_terminating_the_group_leaves_the_result_queue_readable():
    # mp.Manager()'s process is a sibling in *our* group, not a member of the
    # child's, so the kill must not reach it: the PPP timeout path drains this
    # queue afterwards to report the partial plan.
    manager = mp.Manager()
    queue = manager.Queue()
    proc = start_in_process_group(
        _put_then_sleep, args=(queue, "partial plan"), name="test-producer"
    )

    try:
        assert _wait_until(lambda: not queue.empty()), "child never queued anything"

        terminate_process_group(proc)

        assert queue.get_nowait() == "partial plan"
    finally:
        terminate_process_group(proc)
        manager.shutdown()
