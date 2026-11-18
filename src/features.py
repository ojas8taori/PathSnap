"""Candidate road-arc generation and inference-time feature extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .data import Trip

EARTH_M = 6_371_008.8
FEATURE_NAMES = [
    "distance_to_arc_m", "projection_fraction", "arc_length_m", "heading_cosine",
    "heading_sine", "speed_mps", "time_gap_seconds", "previous_step_m",
    "next_step_m", "distance_rank", "from_node_degree", "to_node_degree",
]


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_M * math.asin(min(1.0, math.sqrt(h)))


def _xyz(lonlat: np.ndarray) -> np.ndarray:
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _local_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float, float, float]:
    """Return cross-track metres, clamped projection, length and segment bearing."""
    lat = math.radians(float(point[1]))
    sx = math.radians(float(start[0] - point[0])) * EARTH_M * math.cos(lat)
    sy = math.radians(float(start[1] - point[1])) * EARTH_M
    ex = math.radians(float(end[0] - point[0])) * EARTH_M * math.cos(lat)
    ey = math.radians(float(end[1] - point[1])) * EARTH_M
    vx, vy = ex - sx, ey - sy
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0 else min(1.0, max(0.0, -(sx * vx + sy * vy) / length_sq))
    dx, dy = sx + t * vx, sy + t * vy
    return math.hypot(dx, dy), t, math.sqrt(length_sq), math.atan2(vx, vy)


@dataclass
class CandidateBatch:
    features: np.ndarray
    candidate_arc_ids: list[np.ndarray]
    labels: np.ndarray | None
    candidate_miss_rate: float | None
    reference_arc_ids: np.ndarray | None = None


class CandidateGenerator:
    """Use spherical midpoint search followed by local segment-distance ranking."""

    def __init__(self, radius_m: float = 80.0, top_k: int = 8, index_factor: int = 8):
        self.radius_m = float(radius_m)
        self.top_k = int(top_k)
        self.index_factor = max(2, int(index_factor))

    def build(self, trip: Trip, with_labels: bool = False) -> CandidateBatch:
        points = trip.track[:, :2]
        if not len(points):
            return CandidateBatch(np.empty((0, len(FEATURE_NAMES))), [], None, None, None)
        endpoints = trip.nodes[trip.arcs] if len(trip.arcs) else np.empty((0, 2, 2))
        if not len(endpoints):
            return CandidateBatch(np.empty((0, len(FEATURE_NAMES))), [np.array([], dtype=int) for _ in points], None, None, None)
        midpoints = (endpoints[:, 0] + endpoints[:, 1]) / 2
        # Average on the sphere to avoid the longitude-wrap error near the date line.
        midpoint_vectors = _xyz(endpoints[:, 0]) + _xyz(endpoints[:, 1])
        norms = np.linalg.norm(midpoint_vectors, axis=1)
        midpoint_vectors[norms > 0] /= norms[norms > 0, None]
        tree = cKDTree(midpoint_vectors)
        point_vectors = _xyz(points)
        query_k = min(len(endpoints), max(self.top_k * self.index_factor, self.top_k))
        _, nearest = tree.query(point_vectors, k=query_k)
        if query_k == 1:
            nearest = nearest[:, None]

        degrees = np.zeros(len(trip.nodes), dtype=float)
        if len(trip.arcs):
            np.add.at(degrees, trip.arcs[:, 0], 1)
            np.add.at(degrees, trip.arcs[:, 1], 1)
        seconds = trip.track[:, 2]
        output_features: list[list[float]] = []
        output_ids: list[np.ndarray] = []
        ref_arcs = self._observation_route_labels(trip, endpoints) if with_labels and trip.route is not None else None
        hit_count = 0
        for i, point in enumerate(points):
            scored = []
            for arc_id in np.atleast_1d(nearest[i]):
                arc_id = int(arc_id)
                distance, fraction, length, bearing = _local_segment(point, endpoints[arc_id, 0], endpoints[arc_id, 1])
                scored.append((distance, arc_id, fraction, length, bearing))
            scored.sort(key=lambda item: (item[0], item[1]))
            within = [row for row in scored if row[0] <= self.radius_m]
            chosen = (within or scored)[: self.top_k]
            chosen.sort(key=lambda item: (item[0], item[1]))
            ids = np.asarray([row[1] for row in chosen], dtype=np.int64)
            output_ids.append(ids)
            prev_dist, next_dist, heading, speed, dt = 0.0, 0.0, 0.0, 0.0, 0.0
            if i > 0:
                prev_dt = max(0.0, float(seconds[i] - seconds[i - 1]))
                prev_dist = haversine(*points[i - 1], *point)
                speed, dt = (prev_dist / prev_dt if prev_dt > 0 else 0.0), prev_dt
                if prev_dist > 0:
                    heading = math.atan2(
                        math.radians(point[0] - points[i - 1, 0]) * math.cos(math.radians(float(point[1]))),
                        math.radians(point[1] - points[i - 1, 1]),
                    )
            if i + 1 < len(points):
                next_dist = haversine(*point, *points[i + 1])
            best_distance = chosen[0][0] if chosen else 0.0
            for rank, (distance, arc_id, fraction, length, bearing) in enumerate(chosen):
                delta = bearing - heading
                output_features.append([
                    distance, fraction, length, math.cos(delta), math.sin(delta), speed, dt,
                    prev_dist, next_dist, float(rank), degrees[int(trip.arcs[arc_id, 0])],
                    degrees[int(trip.arcs[arc_id, 1])],
                ])
            if ref_arcs is not None and int(ref_arcs[i]) in ids:
                hit_count += 1
        labels = None
        miss_rate = None
        if ref_arcs is not None:
            label_values = []
            for i, ids in enumerate(output_ids):
                label_values.extend((ids == ref_arcs[i]).astype(np.int8).tolist())
            labels = np.asarray(label_values, dtype=np.int8)
            miss_rate = 1.0 - hit_count / max(1, len(points))
        return CandidateBatch(np.asarray(output_features, dtype=float), output_ids, labels, miss_rate, ref_arcs)

    @staticmethod
    def _observation_route_labels(trip: Trip, all_endpoints: np.ndarray) -> np.ndarray:
        """Map each GPS observation to the closest arc on its own ordered reference route."""
        if trip.route is None or not len(trip.route):
            return np.full(len(trip.track), -1, dtype=np.int64)
        route_endpoints = all_endpoints[trip.route]
        route_vectors = _xyz(route_endpoints[:, 0]) + _xyz(route_endpoints[:, 1])
        norms = np.linalg.norm(route_vectors, axis=1)
        route_vectors[norms > 0] /= norms[norms > 0, None]
        route_tree = cKDTree(route_vectors)
        nearest_count = min(32, len(route_endpoints))
        nearest_route = route_tree.query(_xyz(trip.track[:, :2]), k=nearest_count)[1]
        if nearest_count == 1:
            nearest_route = nearest_route[:, None]
        chosen = np.empty(len(trip.track), dtype=np.int64)
        for i, point in enumerate(trip.track[:, :2]):
            distances = np.fromiter(
                (_local_segment(point, route_endpoints[int(j), 0], route_endpoints[int(j), 1])[0] for j in np.atleast_1d(nearest_route[i])),
                dtype=float, count=nearest_count,
            )
            chosen[i] = int(trip.route[int(nearest_route[i, int(np.argmin(distances))])])
        return chosen
