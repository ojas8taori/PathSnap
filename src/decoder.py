"""Candidate-arc hidden Markov model and Viterbi decoder."""

from __future__ import annotations

import heapq
import math
from functools import lru_cache

import numpy as np

from .data import Trip
from .features import EARTH_M, haversine


def _edge_length_m(trip: Trip) -> np.ndarray:
    return np.asarray([
        haversine(*trip.nodes[int(u)], *trip.nodes[int(v)]) for u, v in trip.arcs
    ], dtype=float)


def _viterbi_block(
    trip: Trip,
    candidate_ids: list[np.ndarray],
    emissions: list[np.ndarray],
    indices: list[int],
    edge_lengths: np.ndarray,
    max_speed_mps: float,
    sigma_m: float,
    disconnected_log_penalty: float,
) -> list[int]:
    """Decode one contiguous timestamp block, retaining original trip arc IDs."""
    if not indices:
        return []
    score = np.log(np.maximum(emissions[indices[0]], 1e-12))
    backpointers: list[np.ndarray] = []

    @lru_cache(maxsize=100_000)
    def node_distance(source: int, target: int, max_distance: int) -> float:
        if source == target:
            return 0.0
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if distance > max_distance:
                break
            if node == target:
                return distance
            for arc_id in outgoing.get(node, ()):
                next_node = int(trip.arcs[arc_id, 1])
                next_distance = distance + float(edge_lengths[arc_id])
                if next_distance < distances.get(next_node, math.inf) and next_distance <= max_distance:
                    distances[next_node] = next_distance
                    heapq.heappush(queue, (next_distance, next_node))
        return math.inf

    outgoing: dict[int, list[int]] = {}
    for arc_id, (source, _) in enumerate(trip.arcs):
        outgoing.setdefault(int(source), []).append(arc_id)

    for tpos in range(1, len(indices)):
        previous_i, current_i = indices[tpos - 1], indices[tpos]
        previous_ids = candidate_ids[previous_i]
        current_ids = candidate_ids[current_i]
        dt = max(0.0, float(trip.track[current_i, 2] - trip.track[previous_i, 2]))
        observed_m = haversine(*trip.track[previous_i, :2], *trip.track[current_i, :2])
        plausible_limit = max(500.0, max_speed_mps * max(1.0, dt) + 250.0)
        next_score = np.full(len(current_ids), -math.inf)
        pointer = np.zeros(len(current_ids), dtype=np.int64)
        for j, next_arc in enumerate(current_ids):
            next_start = int(trip.arcs[int(next_arc), 0])
            for i, previous_arc in enumerate(previous_ids):
                previous_arc = int(previous_arc)
                if previous_arc == int(next_arc):
                    route_m = 0.0
                else:
                    previous_end = int(trip.arcs[previous_arc, 1])
                    path_m = node_distance(previous_end, next_start, int(plausible_limit))
                    route_m = float(edge_lengths[int(next_arc)]) + path_m if math.isfinite(path_m) else math.inf
                if math.isfinite(route_m):
                    transition = -abs(route_m - observed_m) / max(1.0, sigma_m + 0.25 * observed_m)
                else:
                    transition = disconnected_log_penalty
                value = score[i] + transition
                if value > next_score[j]:
                    next_score[j] = value
                    pointer[j] = i
        next_score += np.log(np.maximum(emissions[current_i], 1e-12))
        backpointers.append(pointer)
        score = next_score
    state = int(np.argmax(score))
    states = [state]
    for pointer in reversed(backpointers):
        state = int(pointer[state])
        states.append(state)
    states.reverse()
    return [int(candidate_ids[index][state_idx]) for index, state_idx in zip(indices, states)]


def viterbi_decode(
    trip: Trip,
    candidate_ids: list[np.ndarray],
    classifier_scores: list[np.ndarray],
    max_gap_seconds: float = 120.0,
    max_speed_mps: float = 45.0,
    sigma_m: float = 80.0,
    disconnected_log_penalty: float = -12.0,
) -> list[int]:
    """Decode arc states using per-observation classifier scores as emissions.

    Scores are converted to a within-observation softmax because score scales may
    differ by estimator; SVM decision values are comparative scores, not claimed
    calibrated probabilities. Long time gaps reset the chain and missing states
    fall back to -1 when no candidate is available.
    """
    if len(candidate_ids) != len(trip.track) or len(classifier_scores) != len(trip.track):
        raise ValueError("one candidate list and classifier score vector are required per GPS observation")
    candidate_ids = [np.asarray(ids, dtype=np.int64) for ids in candidate_ids]
    emissions = []
    for ids, scores in zip(candidate_ids, classifier_scores):
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(ids):
            raise ValueError("candidate IDs and score vector lengths differ")
        if not len(ids):
            emissions.append(np.empty(0))
            continue
        logits = scores - np.max(scores)
        weights = np.exp(np.clip(logits, -60, 0))
        emissions.append(weights / weights.sum())
    output = [-1] * len(trip.track)
    edge_lengths = _edge_length_m(trip)
    valid = [i for i, ids in enumerate(candidate_ids) if len(ids)]
    if not valid:
        return output
    blocks: list[list[int]] = [[valid[0]]]
    for index in valid[1:]:
        previous = blocks[-1][-1]
        gap = float(trip.track[index, 2] - trip.track[previous, 2])
        if index != previous + 1 or gap <= 0 or gap > max_gap_seconds:
            blocks.append([index])
        else:
            blocks[-1].append(index)
    for block in blocks:
        chosen = _viterbi_block(trip, candidate_ids, emissions, block, edge_lengths, max_speed_mps, sigma_m, disconnected_log_penalty)
        for index, arc_id in zip(block, chosen):
            output[index] = arc_id
    return output
