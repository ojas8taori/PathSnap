"""Audit the supplied PathSnap ZIP without modifying or extracting it."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import PurePosixPath

EXPECTED = {"track", "nodes", "arcs", "route"}
PARTITIONS = {"train", "student_test", "evaluator_test"}


def read_rows(archive: zipfile.ZipFile, member: str) -> list[list[float]]:
    text = archive.read(member).decode("utf-8-sig")
    rows: list[list[float]] = []
    for line in io.StringIO(text):
        if line.strip():
            rows.append([float(value) for value in line.split()])
    return rows


def components(node_count: int, arcs: list[list[float]]) -> tuple[int, int]:
    graph = [[] for _ in range(node_count)]
    active: set[int] = set()
    for row in arcs:
        if len(row) != 2:
            continue
        u, v = map(int, row)
        if 0 <= u < node_count and 0 <= v < node_count:
            graph[u].append(v)
            graph[v].append(u)
            active.update((u, v))
    seen: set[int] = set()
    count = 0
    for start in active:
        if start in seen:
            continue
        count += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return count, node_count - len(active)


def audit(path: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = [PurePosixPath(name) for name in archive.namelist()]
        trip_files: dict[str, dict[str, str]] = defaultdict(dict)
        for member in members:
            if len(member.parts) == 4 and member.parts[1] in PARTITIONS:
                trip_files[member.parts[1] + "/" + member.parts[2]][member.suffix[1:]] = str(member)

        results = {"archive": path, "zip_entries": len(members), "partitions": {}}
        for partition in sorted(PARTITIONS):
            trips = sorted(key for key in trip_files if key.startswith(partition + "/"))
            rows_total = Counter()
            bytes_total = Counter()
            errors = Counter()
            trip_summaries = []
            all_lons: list[float] = []
            all_lats: list[float] = []
            intervals: list[float] = []
            counts = {"observations": [], "nodes": [], "arcs": [], "route": []}
            route_ref_invalid = 0
            route_transitions = 0
            route_disconnected_transitions = 0
            arc_endpoint_invalid = 0
            isolated_nodes = 0
            component_counts = []

            for key in trips:
                files = trip_files[key]
                missing = sorted(EXPECTED - set(files))
                if missing:
                    errors["missing_files"] += len(missing)
                    continue
                data = {ext: read_rows(archive, files[ext]) for ext in EXPECTED}
                for ext, values in data.items():
                    rows_total[ext] += len(values)
                    bytes_total[ext] += archive.getinfo(files[ext]).file_size
                track, nodes, arcs, route = (data[x] for x in ("track", "nodes", "arcs", "route"))
                counts["observations"].append(len(track))
                counts["nodes"].append(len(nodes))
                counts["arcs"].append(len(arcs))
                counts["route"].append(len(route))
                for row in track:
                    if len(row) != 3:
                        errors["track_wrong_width"] += 1
                    else:
                        lon, lat, _ = row
                        if not all(math.isfinite(value) for value in row):
                            errors["track_non_finite_values"] += 1
                        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                            errors["gps_coordinates_out_of_range"] += 1
                        all_lons.append(lon)
                        all_lats.append(lat)
                for row in nodes:
                    if len(row) != 2:
                        errors["node_wrong_width"] += 1
                    else:
                        if not all(math.isfinite(value) for value in row):
                            errors["node_non_finite_values"] += 1
                        all_lons.append(row[0])
                        all_lats.append(row[1])
                timestamps = [row[2] for row in track if len(row) == 3]
                dt = [b - a for a, b in zip(timestamps, timestamps[1:])]
                intervals.extend(dt)
                errors["nonpositive_timestamp_steps"] += sum(value <= 0 for value in dt)
                errors["timestamp_steps_over_10s"] += sum(value > 10 for value in dt)
                errors["timestamp_steps_not_1s"] += sum(value != 1 for value in dt)
                errors["duplicate_track_rows"] += sum(track[i] == track[i - 1] for i in range(1, len(track)))
                for row in arcs:
                    if len(row) != 2 or any(value != int(value) for value in row):
                        errors["invalid_arc_schema"] += 1
                    elif any(value < 0 or value >= len(nodes) for value in row):
                        arc_endpoint_invalid += 1
                for row in route:
                    if len(row) != 1 or row[0] != int(row[0]):
                        errors["invalid_route_schema"] += 1
                    elif row[0] < 0 or row[0] >= len(arcs):
                        route_ref_invalid += 1
                route_ids = [int(row[0]) for row in route if len(row) == 1 and row[0] == int(row[0]) and 0 <= row[0] < len(arcs)]
                for previous, current in zip(route_ids, route_ids[1:]):
                    if len(arcs[previous]) == 2 and len(arcs[current]) == 2:
                        route_transitions += 1
                        if int(arcs[previous][1]) != int(arcs[current][0]):
                            route_disconnected_transitions += 1
                comp_count, isolated = components(len(nodes), arcs)
                component_counts.append(comp_count)
                isolated_nodes += isolated
                trip_summaries.append({
                    "trip_id": key.split("/", 1)[1],
                    "observations": len(track), "nodes": len(nodes), "arcs": len(arcs),
                    "route_arcs": len(route), "largest_time_gap_seconds": max(dt, default=None),
                    "graph_undirected_components": comp_count,
                })

            def summary(values: list[int | float]) -> dict:
                if not values:
                    return {"count": 0}
                ordered = sorted(values)
                return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}

            results["partitions"][partition] = {
                "trip_count": len(trips), "total_rows_by_extension": dict(rows_total),
                "uncompressed_bytes_by_extension": dict(bytes_total),
                "missing_file_trips": sum(1 for key in trips if set(trip_files[key]) != EXPECTED),
                "row_count_summaries": {name: summary(values) for name, values in counts.items()},
                "timestamp_interval_seconds": summary(intervals),
                "timestamp_steps_not_1s": errors["timestamp_steps_not_1s"],
                "timestamp_steps_over_10s": errors["timestamp_steps_over_10s"],
                "nonpositive_timestamp_steps": errors["nonpositive_timestamp_steps"],
                "coordinate_bounds_lon_lat": {
                    "lon_min": min(all_lons) if all_lons else None, "lon_max": max(all_lons) if all_lons else None,
                    "lat_min": min(all_lats) if all_lats else None, "lat_max": max(all_lats) if all_lats else None,
                },
                "invalid_schema_counts": dict(errors),
                "invalid_arc_endpoints": arc_endpoint_invalid, "invalid_route_arc_ids": route_ref_invalid,
                "route_arc_transitions": route_transitions,
                "route_disconnected_transitions": route_disconnected_transitions,
                "undirected_component_count": summary(component_counts),
                "isolated_node_count_total": isolated_nodes,
                "trips": trip_summaries,
            }
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to Student folder.zip")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    result = audit(args.archive)
    payload = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
