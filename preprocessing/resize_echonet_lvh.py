"""
Resize EchoNet-LVH videos to a small square frame size once, on disk.

EchoNetLVHDataset decodes every full-resolution (e.g., 768x1024) frame of a video on each
__getitem__ call and then resizes the sampled clip. Resizing all videos ahead of time makes
decoding much cheaper. The output mirrors the input layout (Batch*/<HashedFileName>.avi plus
MeasurementsList.csv), so the dataset can be pointed at the output directory unchanged.

Frames are resized independently with cv2.INTER_AREA, exactly as EchoNetLVHDataset._resize
does, so resizing before clip sampling gives the same frames as resizing after it. With the
default lossless FFV1 codec, the stored frames are bit-identical to the on-the-fly result;
every written file is read back and checked unless --no_verify is given.

Usage:
    python preprocessing/resize_echonet_lvh.py --data_dir data --out_dir data_112
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed

LOSSLESS_CODECS = {"FFV1"}
DEFAULT_FPS = 30.0


def read_resized(path, frame_size):
    """Decode a video frame by frame, resizing each frame as it is read."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"cannot open {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    n_meta = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(
            cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        )
    capture.release()

    if not frames:
        raise IOError(f"no frames decoded from {path}")

    return np.stack(frames, axis=0), fps, n_meta, width, height


def read_all(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return np.stack(frames, axis=0) if frames else np.empty((0,))


def write_video(path, video, fps, codec):
    height, width = video.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height), True
    )
    if not writer.isOpened():
        raise IOError(f"cannot open VideoWriter with codec {codec!r} for {path}")
    for frame in video:
        writer.write(frame)
    writer.release()


def process_one(src, dst, partial_dir, frame_size, codec, verify, overwrite):
    record = {
        "HashedFileName": src.stem,
        "batch": src.parent.name,
        "src_path": str(src),
        "dst_path": str(dst),
        "status": None,
        "src_width": None,
        "src_height": None,
        "fps": None,
        "frames_meta": None,
        "frames_decoded": None,
        "max_abs_diff": None,
        "error": None,
    }

    if dst.exists() and not overwrite:
        record["status"] = "skipped_exists"
        return record

    tmp = partial_dir / f"{src.parent.name}__{src.name}"
    try:
        cv2.setNumThreads(1)  # avoid oversubscription across joblib workers
        video, fps, n_meta, width, height = read_resized(src, frame_size)
        fps = fps if fps and fps > 0 else DEFAULT_FPS
        record.update(
            src_width=width,
            src_height=height,
            fps=fps,
            frames_meta=n_meta,
            frames_decoded=len(video),
        )

        write_video(tmp, video, fps, codec)

        if verify:
            back = read_all(tmp)
            if back.shape != video.shape:
                raise ValueError(f"read-back shape {back.shape} != written {video.shape}")
            diff = int(np.abs(back.astype(np.int16) - video.astype(np.int16)).max())
            record["max_abs_diff"] = diff
            if codec in LOSSLESS_CODECS and diff != 0:
                raise ValueError(f"lossless codec {codec} changed pixels (max diff {diff})")

        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)  # publish only complete, verified files
        record["status"] = "ok"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def main(args):
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir == data_dir:
        sys.exit("ERROR: --out_dir must differ from --data_dir (originals would be overwritten)")

    labels = data_dir / "MeasurementsList.csv"
    if not labels.is_file():
        sys.exit(f"ERROR: {labels} not found")

    batch_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Batch")
    )
    tasks = [
        (src, out_dir / batch_dir.name / src.name)
        for batch_dir in batch_dirs
        for src in sorted(batch_dir.glob("*.avi"))
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit(f"ERROR: no .avi files found under {data_dir}/Batch*/")

    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / ".partial"
    partial_dir.mkdir(exist_ok=True)
    shutil.copy2(labels, out_dir / "MeasurementsList.csv")

    print(
        f"Resizing {len(tasks)} videos from {len(batch_dirs)} batch folder(s) to "
        f"{args.frame_size}x{args.frame_size} with {args.codec} -> {out_dir}"
    )

    results = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(process_one)(
            src,
            dst,
            partial_dir,
            args.frame_size,
            args.codec,
            not args.no_verify,
            args.overwrite,
        )
        for src, dst in tasks
    )
    records = list(tqdm.tqdm(results, total=len(tasks), desc="Resizing"))

    manifest = pd.DataFrame(records)
    manifest_path = out_dir / "resize_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if not any(partial_dir.iterdir()):
        partial_dir.rmdir()

    counts = manifest["status"].value_counts().to_dict()
    print(f"Done: {counts}. Manifest: {manifest_path}")

    mismatched = manifest[
        (manifest["status"] == "ok")
        & (manifest["frames_meta"] != manifest["frames_decoded"])
    ]
    if len(mismatched):
        print(
            f"Note: {len(mismatched)} video(s) decoded a different number of frames than "
            "their header reports (see frames_meta vs frames_decoded in the manifest)."
        )

    failed = manifest[manifest["status"] == "failed"]
    if len(failed):
        print(f"{len(failed)} video(s) failed; first errors:")
        for _, row in failed.head(5).iterrows():
            print(f"  {row['HashedFileName']}: {row['error']}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=str, required=True, help="EchoNet-LVH root (Batch*/ + MeasurementsList.csv)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root, mirrors the input layout")
    parser.add_argument("--frame_size", type=int, default=112)
    parser.add_argument("--codec", type=str, default="FFV1", choices=["FFV1", "MJPG"], help="FFV1 is lossless; MJPG is lossy but smaller")
    parser.add_argument("--n_jobs", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos (for testing)")
    parser.add_argument("--overwrite", action="store_true", help="Re-process videos that already exist in --out_dir")
    parser.add_argument("--no_verify", action="store_true", help="Skip reading back each written file")

    main(parser.parse_args())
