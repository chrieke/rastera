"""Unit tests for rastera/config.py: setter validation and bounded gather."""

import asyncio

import pytest

import rastera
from rastera import config


@pytest.fixture(autouse=True)
def _reset_concurrency():
    yield
    rastera.set_concurrency(merge=1, vrt=1, dimap=1)


class TestSetConcurrencyValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, "x", True, False])
    def test_rejects_invalid(self, bad):
        # bool is an int subclass in Python, so we explicitly reject it
        # to avoid `set_concurrency(merge=True)` silently meaning n=1.
        with pytest.raises(ValueError, match="must be int >= 1"):
            rastera.set_concurrency(merge=bad)

    def test_accepts_int_ge_one(self):
        rastera.set_concurrency(merge=1)
        rastera.set_concurrency(merge=2, vrt=4, dimap=8)
        assert config._merge_concurrency == 2
        assert config._vrt_concurrency == 4
        assert config._dimap_concurrency == 8

    def test_partial_update(self):
        rastera.set_concurrency(merge=4)
        assert config._merge_concurrency == 4
        assert config._vrt_concurrency == 1
        assert config._dimap_concurrency == 1
        rastera.set_concurrency(vrt=3)
        assert config._merge_concurrency == 4
        assert config._vrt_concurrency == 3
        assert config._dimap_concurrency == 1

    def test_none_is_noop(self):
        rastera.set_concurrency(merge=5)
        rastera.set_concurrency(merge=None)
        assert config._merge_concurrency == 5


class TestGatherBounded:
    async def test_preserves_input_order(self):
        """Coros that resolve in reverse start order still return in input order."""

        async def make_coro(i, delay):
            await asyncio.sleep(delay)
            return i

        # Index 0 sleeps longest, index N-1 sleeps shortest — completion order
        # is the reverse of input order.
        coros = [make_coro(i, 0.05 - i * 0.01) for i in range(5)]
        results = await config._gather_bounded(8, coros)
        assert results == [0, 1, 2, 3, 4]

    async def test_n_one_runs_sequentially(self):
        """At n=1 the helper short-circuits to sequential awaits."""
        in_flight = 0
        max_in_flight = 0

        async def make_coro():
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return None

        coros = [make_coro() for _ in range(5)]
        await config._gather_bounded(1, coros)
        assert max_in_flight == 1

    async def test_n_caps_in_flight(self):
        in_flight = 0
        max_in_flight = 0

        async def make_coro():
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return None

        coros = [make_coro() for _ in range(20)]
        await config._gather_bounded(4, coros)
        assert max_in_flight == 4

    async def test_empty_input(self):
        results = await config._gather_bounded(4, [])
        assert results == []
