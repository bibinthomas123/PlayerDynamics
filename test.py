# scripts/player_validation_audit.py

import argparse
import pandas as pd
import numpy as np


def tracking_coverage(df, players):
    print("\n=== Tracking coverage ===")

    for pid, label in players.items():
        p = df[df["mapped id"] == pid]

        if p.empty:
            print(f"{label:<12}: NOT FOUND")
            continue

        span = p["elapsed_s"].max() - p["elapsed_s"].min()
        expected = span * 20
        actual = len(p)

        coverage = actual / expected * 100 if expected > 0 else 0

        spd_dist = (p["speed"].clip(0, 12) * 0.05).sum() / 1000

        p2 = p.dropna(subset=["x", "y"]).sort_values("elapsed_s")

        euc_dist = (
            ((p2["x"].diff() ** 2 + p2["y"].diff() ** 2) ** 0.5).sum()
            / 1000
        )

        corrected = (
            spd_dist / (coverage / 100)
            if coverage > 0
            else spd_dist
        )

        print(
            f"{label:<12}: "
            f"coverage={coverage:.0f}% "
            f"speed*dt={spd_dist:.2f}km "
            f"euclidean={euc_dist:.2f}km "
            f"est_true={corrected:.2f}km"
        )


def locomotor_audit(df, threshold):
    print("\n=== Locomotor Suppression Audit ===")
    print(f"Current threshold = {threshold:.2f} m/s")

    for pid, label in {
        2058: "Wing LA",
        1164: "Pivot KR",
        2331: "GK TW",
    }.items():

        p = (
            df[df["mapped id"] == pid]
            .sort_values("elapsed_s")
        )

        if p.empty:
            continue

        step = 15
        avgs = []

        t = p["elapsed_s"].min()
        tmax = p["elapsed_s"].max()

        while t + step <= tmax:

            seg = p[
                (p["elapsed_s"] >= t)
                & (p["elapsed_s"] < t + step)
            ]

            if len(seg) > 3:
                avgs.append(seg["speed"].clip(0, 12).mean())

            t += step

        avgs = np.array(avgs)

        print(
            f"{label:<12}: "
            f"median={np.median(avgs):.2f} "
            f"below={((avgs <= threshold).mean()*100):.0f}%"
        )


def zone_radius_audit():
    print("\n=== Zone Radius Audit ===")

    zone_r_pitch = 5.0

    x_scale = 40 / 100
    y_scale = 20 / 100

    print(
        f"Current radius = {zone_r_pitch} pitch units"
    )

    print(
        f"Effective radius X = {zone_r_pitch*x_scale:.2f}m"
    )

    print(
        f"Effective radius Y = {zone_r_pitch*y_scale:.2f}m"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Handball Analytics Validation Audit"
    )

    parser.add_argument(
        "--cache",
        default="data/_analysis_cache.pkl",
        help="Path to cached dataframe"
    )

    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=2.5,
        help="Current locomotor suppression threshold"
    )

    args = parser.parse_args()

    print(f"Loading: {args.cache}")

    df = pd.read_pickle(args.cache)

    players = {
        2058: "Wing LA",
        1164: "Pivot KR",
        2331: "GK TW",
        2059: "Back RM",
        2261: "Pivot2 KR",
        2407: "GK2 TW",
    }

    tracking_coverage(df, players)
    locomotor_audit(df, args.speed_threshold)
    zone_radius_audit()


if __name__ == "__main__":
    main()