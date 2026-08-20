import multiprocessing as mp

from pfs_target_uploader.utils.ppp import drain_ppp_queue


def test_drain_ppp_queue_returns_last_item_pushed():
    """PPPrunStart pushes one snapshot per pointing plus a final complete
    snapshot; callers must end up with the last one, not the first."""
    queue = mp.Manager().Queue()
    queue.put("first pointing snapshot")
    queue.put("second pointing snapshot")
    queue.put("final complete plan")

    result = drain_ppp_queue(queue)

    assert result == "final complete plan"


def test_drain_ppp_queue_returns_none_for_empty_queue():
    queue = mp.Manager().Queue()

    result = drain_ppp_queue(queue)

    assert result is None


def test_drain_ppp_queue_returns_only_item_for_single_push():
    queue = mp.Manager().Queue()
    queue.put("only snapshot")

    result = drain_ppp_queue(queue)

    assert result == "only snapshot"
