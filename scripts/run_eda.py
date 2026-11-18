"""Create EDA plots and a real Folium map from an actual supplied trip."""

from __future__ import annotations

import sys
import os
from pathlib import Path as _RootPath
sys.path.insert(0, str(_RootPath(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(_RootPath("results/matplotlib-config").resolve()))
_RootPath("results/matplotlib-config").mkdir(parents=True, exist_ok=True)

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker

from src.data import load_trip, trip_ids


def make_map(archive_path: str, partition: str, trip_id: str, output: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        trip = load_trip(archive, partition, trip_id)
    gps = trip.track[:, :2]
    center = [float(np.mean(gps[:, 1])), float(np.mean(gps[:, 0]))]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    gps_group = FeatureGroup(name="GPS observations", show=True)
    points = [[float(lat), float(lon)] for lon, lat in gps]
    PolyLine(points, color="#2474a6", weight=3, opacity=0.65, tooltip="Observed GPS trace").add_to(gps_group)
    for lat, lon in points[:: max(1, len(points) // 1200)]:
        CircleMarker([lat, lon], radius=2, color="#2474a6", fill=True, fill_opacity=0.75).add_to(gps_group)
    gps_group.add_to(fmap)

    margin = 0.01
    west, east = gps[:, 0].min() - margin, gps[:, 0].max() + margin
    south, north = gps[:, 1].min() - margin, gps[:, 1].max() + margin
    arc_group = FeatureGroup(name="Nearby road arcs", show=True)
    nearby = 0
    for u, v in trip.arcs:
        a, b = trip.nodes[int(u)], trip.nodes[int(v)]
        if west <= a[0] <= east and south <= a[1] <= north and west <= b[0] <= east and south <= b[1] <= north:
            PolyLine([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]], color="#8a9aa5", weight=1, opacity=0.45).add_to(arc_group)
            nearby += 1
    arc_group.add_to(fmap)
    if trip.route is not None:
        route_group = FeatureGroup(name="Supplied reference route", show=True)
        for arc_id in trip.route:
            u, v = trip.arcs[int(arc_id)]
            a, b = trip.nodes[int(u)], trip.nodes[int(v)]
            PolyLine([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]], color="#29945e", weight=4, opacity=0.8).add_to(route_group)
        route_group.add_to(fmap)
    LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(gps[:, 1].min()), float(gps[:, 0].min())], [float(gps[:, 1].max()), float(gps[:, 0].max())]])
    fmap.save(str(output))
    print(f"Map saved: {output} (trip={trip_id}, nearby_arcs={nearby}, observations={len(gps)})")


def make_charts(audit_path: Path, output_dir: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train = audit["partitions"]["train"]
    student = audit["partitions"]["student_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, part, color in (("Train", train, "#286b9c"), ("Student test", student, "#c35a3c")):
        times = [trip["largest_time_gap_seconds"] for trip in part["trips"] if trip["largest_time_gap_seconds"] is not None]
        axes[0].hist(np.log10(np.maximum(times, 1)), bins=18, alpha=0.55, label=label, color=color)
    axes[0].set_title("Largest within-trip time gap")
    axes[0].set_xlabel("log10(seconds)")
    axes[0].set_ylabel("Trips")
    axes[0].legend()
    for label, part, color in (("Train", train, "#286b9c"), ("Student test", student, "#c35a3c")):
        values = [trip["observations"] for trip in part["trips"]]
        axes[1].hist(values, bins=18, alpha=0.55, label=label, color=color)
    axes[1].set_title("GPS observations per trip")
    axes[1].set_xlabel("Observations")
    axes[1].set_ylabel("Trips")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trip_and_sampling_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--audit", default="results/dataset_audit.json")
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--trip-id", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    vis = Path("visualizations")
    vis.mkdir(exist_ok=True)
    trip_id = args.trip_id or trip_ids(config["data_archive"], args.partition)[0]
    make_map(config["data_archive"], args.partition, trip_id, vis / "representative_trip.html")
    make_charts(Path(args.audit), Path("results"))


if __name__ == "__main__":
    main()
