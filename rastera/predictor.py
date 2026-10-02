"""Undo the predictor of band-interleaved tiles per band.

async-tiff 0.7.2 joins the bands of a band-interleaved tile into one buffer and
undoes the predictor over it as if the tile were pixel-interleaved: over rows
of ``tile_width * bands`` samples, each summed with the one ``bands`` samples
back. The writer applied it per band, row by row. Single-band files come
out right; with two or more bands nearly every value is wrong, for predictor 2
and 3 alike.

Both undos are lossless, so the stored samples can be recovered from
async-tiff's output and undone again per band.

TODO: remove once async-tiff undoes the predictor per band
(async-tiff PR 340); the steps are next to the ``async-tiff`` pin
in pyproject.toml.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from async_geotiff import GeoTIFF, Overview, RasterArray, Tile, Window
from async_geotiff._read import read
from async_tiff.enums import PlanarConfiguration, Predictor

_Redo = Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]


async def read_window(readable: GeoTIFF | Overview, window: Window) -> RasterArray:
    """``readable.read(window=window)``, with band-interleaved tiles undone
    per band."""
    ifd = readable.ifd
    redo = _REDO.get(ifd.predictor) if ifd.predictor is not None else None
    if (
        redo is None
        or ifd.planar_configuration != PlanarConfiguration.Planar
        or ifd.samples_per_pixel == 1
    ):
        return await readable.read(window=window)
    # _Redone has the rest of what read() takes through __getattr__.
    return await read(_Redone(readable, redo), window=window)  # type: ignore[reportArgumentType]


class _Redone:
    """*readable*, whose tiles come with *redo* applied, for async_geotiff's
    window read to stitch. It reads full tiles, which the redo needs: an edge
    tile is stored, and predicted, at the full tile size."""

    def __init__(self, readable: GeoTIFF | Overview, redo: _Redo) -> None:
        self._readable = readable
        self._redo = redo

    def __getattr__(self, name: str) -> Any:
        return getattr(self._readable, name)

    async def fetch_tiles(self, xy: Sequence[tuple[int, int]]) -> list[Tile]:
        tiles = await self._readable.fetch_tiles(xy)
        # In parallel, as async-tiff decodes them: NumPy releases the GIL.
        data = await asyncio.gather(
            *(asyncio.to_thread(self._redo, t.array.data) for t in tiles)  # type: ignore[reportUnknownMemberType]
        )
        return [
            replace(t, array=replace(t.array, data=d))
            for t, d in zip(tiles, data, strict=True)
        ]


def _redo_horizontal(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Predictor 2: each sample is stored as its difference to the previous
    one in the row."""
    bands, height, width = data.shape
    # Unsigned, so the sums wrap as the predictor's do, for any sample type.
    summed = np.ascontiguousarray(data).view(f"u{data.dtype.itemsize}")
    # async-tiff summed rows of width * bands samples with a stride of bands,
    # which is axis 1 here.
    summed = summed.reshape(height, width, bands)
    stored = summed.copy()
    stored[:, 1:] -= summed[:, :-1]
    stored = stored.reshape(bands, height, width)
    return stored.cumsum(axis=2, dtype=stored.dtype).view(data.dtype)


def _redo_floating_point(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Predictor 3: each row's samples are split into byte planes, most
    significant first, and each byte stored as its difference to the previous
    one."""
    bands, height, width = data.shape
    size = data.dtype.itemsize
    big_endian = data.dtype.newbyteorder(">")
    # async-tiff's rows, as the bytes it summed with a stride of ``bands``.
    planes = np.ascontiguousarray(data, dtype=big_endian).view(np.uint8)
    summed = planes.reshape(height, width * bands, size).transpose(0, 2, 1)
    summed = summed.reshape(height, size * width * bands)
    stored = summed.copy()
    stored[:, bands:] -= summed[:, :-bands]
    # Each band's rows, summed with a stride of 1.
    rows = stored.reshape(bands * height, size * width).cumsum(axis=1, dtype=np.uint8)
    pixels = np.ascontiguousarray(rows.reshape(-1, size, width).transpose(0, 2, 1))
    return pixels.view(big_endian).reshape(bands, height, width).astype(data.dtype)


_REDO: dict[Predictor, _Redo] = {
    Predictor.Horizontal: _redo_horizontal,
    Predictor.FloatingPoint: _redo_floating_point,
}
