"""Runtime locks survive a scoring call abandoned in an earlier event loop."""

import asyncio

from verifiers.v1.runtimes.base import LoopLocks


def test_a_lock_held_by_an_abandoned_loop_does_not_block_a_new_loop() -> None:
    locks = LoopLocks()

    async def abandon() -> None:
        # Hold the lock and let the loop close underneath it, as a timed-out
        # scoring task does when a collection's loop is torn down.
        await locks.get("script").acquire()

    asyncio.run(abandon())

    async def reuse() -> bool:
        async with locks.get("script"):
            return True

    assert asyncio.run(reuse())


def test_locks_are_shared_within_one_loop() -> None:
    locks = LoopLocks()

    async def same() -> bool:
        return locks.get("a") is locks.get("a") and locks.get("a") is not locks.get("b")

    assert asyncio.run(same())
