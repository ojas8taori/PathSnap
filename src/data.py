"""Read trip files from the supplied ZIP archive without modifying it."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray  # longitude, latitude, timestamp seconds
    nodes: np.ndarray  # longitude, latitude; row index is node_id
    arcs: np.ndarray  # from_node_id, to_node_id; row index is arc_id
    route: np.ndarray | None  # chronological arc ids; None if not supplied


def _array(archive: zipfile.ZipFile, name: str, columns: int | None = None) -> np.ndarray:
    raw = archive.read(name).decode("utf-8-sig")
    rows = [[float(v) for v in line.split()] for line in io.StringIO(raw) if line.strip()]
    if not rows:
        return np.empty((0, columns or 0), dtype=np.float64)
    array = np.asarray(rows, dtype=np.float64)
    if columns is not None and (array.ndim != 2 or array.shape[1] != columns):
        raise ValueError(f"{name}: expected {columns} columns, got shape {array.shape}")
    return array


def trip_ids(archive_path: str, partition: str) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        found = set()
        for raw in archive.namelist():
            path = PurePosixPath(raw)
            if len(path.parts) == 4 and path.parts[1] == partition and path.suffix == ".track":
                found.add(path.parts[2])
        return sorted(found)


def load_trip(archive: zipfile.ZipFile, partition: str, trip_id: str) -> Trip:
    base = f"Student folder/{partition}/{trip_id}/{trip_id}"
    track = _array(archive, base + ".track", 3)
    nodes = _array(archive, base + ".nodes", 2)
    arcs_raw = _array(archive, base + ".arcs", 2)
    route_raw = _array(archive, base + ".route") if base + ".route" in archive.namelist() else None
    if arcs_raw.size and not np.equal(arcs_raw, np.floor(arcs_raw)).all():
        raise ValueError(f"{trip_id}: arc endpoints are not integers")
    arcs = arcs_raw.astype(np.int64, copy=False)
    route = None
    if route_raw is not None:
        if route_raw.size and (route_raw.ndim != 2 or route_raw.shape[1] != 1):
            raise ValueError(f"{trip_id}: route must have one column")
        route = route_raw.reshape(-1).astype(np.int64, copy=False)
    if arcs.size and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"{trip_id}: arc endpoint is outside the node table")
    if route is not None and route.size and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"{trip_id}: route arc ID is outside the arc table")
    return Trip(trip_id, partition, track, nodes, arcs, route)


def load_partition(archive_path: str, partition: str) -> list[Trip]:
    with zipfile.ZipFile(archive_path) as archive:
        return [load_trip(archive, partition, trip_id) for trip_id in trip_ids(archive_path, partition)]
