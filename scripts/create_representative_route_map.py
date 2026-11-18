"""Create an interactive Folium map and static report figure for one student trip."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_trip
from src.inference import PathSnapEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="C:/OJ/sem5/ML/hackathon/Student folder.zip")
    parser.add_argument("--trip-id", default="00000007")
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--artifact", default="artifacts/final_model.joblib")
    parser.add_argument("--metadata", default="artifacts/final_metadata.json")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    engine = PathSnapEngine(args.archive, args.artifact, args.partition)
    prediction = engine.predict(args.trip_id)
    with zipfile.ZipFile(args.archive) as archive:
        trip = load_trip(archive, args.partition, args.trip_id)

    gps_lonlat = trip.track[:, :2]
    gps_latlon = [[float(lat), float(lon)] for lon, lat in gps_lonlat]
    predicted_latlon = prediction["predicted_path"]
    reference_latlon = []
    for arc_id in trip.route if trip.route is not None else []:
        u, v = trip.arcs[int(arc_id)]
        a, b = trip.nodes[int(u)], trip.nodes[int(v)]
        reference_latlon.append([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]])

    # Restrict the context network to the actual trip's GPS bounds plus a small margin.
    margin = 0.012
    west, east = float(gps_lonlat[:, 0].min() - margin), float(gps_lonlat[:, 0].max() + margin)
    south, north = float(gps_lonlat[:, 1].min() - margin), float(gps_lonlat[:, 1].max() + margin)
    network = []
    for u, v in trip.arcs:
        a, b = trip.nodes[int(u)], trip.nodes[int(v)]
        if west <= a[0] <= east and south <= a[1] <= north and west <= b[0] <= east and south <= b[1] <= north:
            network.append([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]])

    center = [float(np.mean(gps_lonlat[:, 1])), float(np.mean(gps_lonlat[:, 0]))]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(gps_latlon, color="#2474a6", weight=3, opacity=0.75, tooltip="Observed GPS trace").add_to(gps_layer)
    stride = max(1, len(gps_latlon) // 250)
    for index in range(0, len(gps_latlon), stride):
        folium.CircleMarker(gps_latlon[index], radius=2, color="#2474a6", fill=True, fill_opacity=0.7).add_to(gps_layer)
    gps_layer.add_to(fmap)

    network_layer = folium.FeatureGroup(name="Nearby road network", show=True)
    for segment in network:
        folium.PolyLine(segment, color="#8a9aa5", weight=1, opacity=0.35).add_to(network_layer)
    network_layer.add_to(fmap)

    reference_layer = folium.FeatureGroup(name="Supplied reference route", show=True)
    for segment in reference_latlon:
        folium.PolyLine(segment, color="#29945e", weight=4, opacity=0.8).add_to(reference_layer)
    reference_layer.add_to(fmap)

    predicted_layer = folium.FeatureGroup(name="HGB + HMM/Viterbi prediction", show=True)
    for segment in predicted_latlon:
        if segment:
            folium.PolyLine(segment, color="#c23c32", weight=3, opacity=0.9).add_to(predicted_layer)
    predicted_layer.add_to(fmap)
    folium.Marker(gps_latlon[0], tooltip="Start", icon=folium.Icon(color="blue", icon="play")).add_to(fmap)
    folium.Marker(gps_latlon[-1], tooltip="End", icon=folium.Icon(color="black", icon="flag")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(gps_lonlat[:, 1].min()), float(gps_lonlat[:, 0].min())],
                     [float(gps_lonlat[:, 1].max()), float(gps_lonlat[:, 0].max())]])
    legend = """<div style='position:fixed;bottom:26px;left:26px;z-index:9999;background:white;padding:9px 12px;border:1px solid #bbb;font-size:12px'>
    <b>PathSnap trip layers</b><br><span style='color:#2474a6'>━━</span> GPS observations<br>
    <span style='color:#8a9aa5'>━━</span> Nearby road network<br><span style='color:#29945e'>━━</span> Supplied reference route<br>
    <span style='color:#c23c32'>━━</span> HGB + HMM/Viterbi prediction</div>"""
    fmap.get_root().html.add_child(folium.Element(legend))

    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "representative_trip.html"
    png_path = out_dir / "representative_trip.png"
    fmap.save(str(html_path))

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=180)
    for segment in network:
        ax.plot([p[1] for p in segment], [p[0] for p in segment], color="#a9b4bc", linewidth=0.55, alpha=0.5)
    for segment in reference_latlon:
        ax.plot([p[1] for p in segment], [p[0] for p in segment], color="#29945e", linewidth=1.6, alpha=0.8)
    for segment in predicted_latlon:
        if segment:
            ax.plot([p[1] for p in segment], [p[0] for p in segment], color="#c23c32", linewidth=1.25, alpha=0.87)
    ax.plot(gps_lonlat[:, 0], gps_lonlat[:, 1], color="#2474a6", linewidth=0.9, alpha=0.7, label="GPS observations")
    ax.scatter([gps_lonlat[0, 0]], [gps_lonlat[0, 1]], s=18, color="#2474a6", zorder=5, label="Start")
    ax.scatter([gps_lonlat[-1, 0]], [gps_lonlat[-1, 1]], s=18, color="#17212b", zorder=5, label="End")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2474a6", lw=2, label="GPS observations"),
        Line2D([0], [0], color="#a9b4bc", lw=2, label="Nearby road network"),
        Line2D([0], [0], color="#29945e", lw=2, label="Reference route"),
        Line2D([0], [0], color="#c23c32", lw=2, label="HGB + HMM/Viterbi prediction"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=8)
    ax.set_title(f"Student-test trip {args.trip_id}: GPS, network, reference and prediction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {html_path} and {png_path}; nearby arcs={len(network)}; observations={len(gps_latlon)}; HMM metrics={prediction['metrics']}")


if __name__ == "__main__":
    main()
