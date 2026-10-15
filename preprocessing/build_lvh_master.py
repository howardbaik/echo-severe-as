"""
Build a master LVH label file from the raw EchoNet-LVH measurements.

`data/MeasurementsList.csv` is long-format (one row per measurement, with
`Calc`/`CalcValue` columns) and covers all official train/val/test videos.
This script pivots it to one row per video and derives a binary LVH label,
independent of which videos happen to be downloaded locally -- it is meant
to be the canonical source that `train.csv`/`val.csv`/`test.csv` are in turn
filtered from (by `split`, and then by local video availability).

LVH is defined by sex-agnostic wall thickness per ASE/EACVI 2015 chamber
quantification guidelines: max(IVSd, LVPWd) >= threshold_cm. The default of
1.1 cm exceeds the upper limit of normal for both men (1.0 cm) and women
(0.9 cm). Duplicate measurements of the same Calc (per video) are aggregated
by median, and videos missing any of IVSd, LVPWd, or LVIDd are dropped.

Usage:
    python preprocessing/build_lvh_master.py --data_dir data
"""

import argparse
import os

import pandas as pd


def build_lvh_master(csv_path, threshold_cm=1.1):
    """
    Pivot MeasurementsList.csv to one row per video and derive an LVH label.

    Parameters
    ----------
    csv_path : str
        Path to MeasurementsList.csv (measurements in cm)
    threshold_cm : float
        Wall thickness cutoff (cm) at or above which a video is labeled LVH

    Returns
    -------
    pd.DataFrame
        One row per video with columns HashedFileName, split, IVSd, LVIDd,
        LVPWd, max_wall, lvh (0/1)
    """
    df = pd.read_csv(csv_path)
    df = df[df["Calc"].isin(["IVSd", "LVPWd", "LVIDd"])]

    wide = (
        df.groupby(["HashedFileName", "split", "Calc"])["CalcValue"]
        .median()
        .unstack("Calc")
        .dropna()
        .reset_index()
    )
    wide.columns.name = None

    wide["max_wall"] = wide[["IVSd", "LVPWd"]].max(axis=1)
    wide["lvh"] = (wide["max_wall"] >= threshold_cm).astype(int)

    return wide[
        ["HashedFileName", "split", "IVSd", "LVIDd", "LVPWd", "max_wall", "lvh"]
    ]


def main(args):
    csv_path = os.path.join(args.data_dir, "MeasurementsList.csv")
    out_path = os.path.join(args.data_dir, "MeasurementsList-LVH.csv")

    master_df = build_lvh_master(csv_path, threshold_cm=args.threshold_cm)
    master_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}: {master_df.shape}")
    print(master_df["split"].value_counts())
    print(master_df["lvh"].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="directory containing MeasurementsList.csv",
    )
    parser.add_argument(
        "--threshold_cm",
        type=float,
        default=1.1,
        help="wall thickness cutoff (cm) at or above which a video is labeled LVH",
    )
    args = parser.parse_args()

    main(args)
