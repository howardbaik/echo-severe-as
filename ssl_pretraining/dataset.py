import itertools
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm

from scipy.ndimage import rotate

from utils import load_video


class EchoDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that loads echocardiogram videos for self-supervised pretraining. This extends the SimCLR framework by selecting two *different* videos from the
    *same* study (patient) as "positive pairs", rather than artifically generating two views of the same sample via augmentation. For each of the two selected videos,
    a clip is randomly subsampled, that clip is lightly augmented in a spatially consistent manner, then the frames of the clip are randomly shuffled (permuted).
    Frame re-ordering is used as an auxiliary pretraining task, so video clip #1, video clip #2, true frame order #1, and true frame order #1 are returned at each iteration.
    The dataset consists of all unique pairs of different videos from the same study.

    Attributes
    ----------
    split: str
        Data split used to select studies that will be loaded (one of ["train", "val", "test", "ext_test"])
    clip_len : int
        Number of frames to form video clips for training (clip length)
    sampling_rate : int
        Temporal "stride" when sampling frames to form clips (e.g., sampling_rate=1 samples consecutive frames)
    video_dir : str
        Path to directory containing videos in .avi format
    label_df : pandas DataFrame
        Data frame containing file names and labels (only file names used here)
    fnames_i : list[str]
        List of paths to "video #1" to be returned at each iteration
    fnames_j : list[str]
        List of paths to "video #2" to be returned at each iteration
    temporal_orderings : list[tuple]
        List of all permutations of frame indices (target classes for frame re-ordering task)

    Methods
    -------
    _sample_frames(x)
        Subsample frames of video to form a video "clip" for training
    _augment(x)
        Apply spatial augmentation to frames of video clip, then randomly shuffle frames
    __len__
        Returns "length"/size of dataset
    __getitem__
        Returns video clip #1, video clip #2, frame order label #1, and frame order label #2 as described above
    """

    def __init__(self, data_dir, split, clip_len=16, sampling_rate=1, n=None):
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate

        self.video_dir = os.path.join(data_dir, "videos")
        self.label_df = pd.read_csv(os.path.join(data_dir, self.split + ".csv"))

        if n is not None:
            self.label_df = self.label_df.iloc[:n, :]

        study_ids = np.sort(self.label_df["acc_num"].unique())

        self.fnames_i = []
        self.fnames_j = []
        for study_id in tqdm.tqdm(study_ids):
            fnames = self.label_df[self.label_df["acc_num"] == study_id][
                "fpath"
            ].values.tolist()

            if len(fnames) == 1:
                self.fnames_i.append(fnames[0])
                self.fnames_j.append(fnames[0])
            else:
                for fname_pair in itertools.combinations(fnames, 2):
                    self.fnames_i.append(fname_pair[0])
                    self.fnames_j.append(fname_pair[1])

        self.temporal_orderings = [
            _ for _ in itertools.permutations(np.arange(self.clip_len))
        ]

    def _sample_frames(self, x):
        if x.shape[0] > self.clip_len * self.sampling_rate:
            start_idx = np.random.choice(
                x.shape[0] - self.clip_len * self.sampling_rate, size=1
            )[0]
            x = x[
                start_idx : (
                    start_idx + self.clip_len * self.sampling_rate
                ) : self.sampling_rate
            ]
        else:
            x = x[:: self.sampling_rate]
            x = np.pad(
                x,
                ((0, self.clip_len - x.shape[0]), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )

        return x

    def _augment(self, x):
        # Zero-pad by up to 8 pixels
        pad = 8

        l, h, w, c = x.shape
        temp = np.zeros((l, h + 2 * pad, w + 2 * pad, c), dtype=x.dtype)
        temp[:, pad:-pad, pad:-pad, :] = x
        i, j = np.random.randint(0, 2 * pad, 2)
        x = temp[:, i : (i + h), j : (j + w), :]

        # Random horizontal flip
        if random.uniform(0, 1) > 0.5:
            x = np.stack([cv2.flip(frame, 1) for frame in x], axis=0)

        # Random rotation between -10 and 10 degrees
        if random.uniform(0, 1) > 0.5:
            angle = np.random.choice(np.arange(-10, 11), size=1)[0]

            x = np.stack([rotate(frame, angle, reshape=False) for frame in x], axis=0)

        # Frame re-ordering
        reordering_label = np.random.choice(len(self.temporal_orderings), size=1)[0]
        reordering = self.temporal_orderings[reordering_label]
        x = x[reordering, :, :, :]

        return x, reordering_label

    def __len__(self):
        return len(self.fnames_i)

    def __getitem__(self, idx):
        x_i = load_video(os.path.join(self.video_dir, self.fnames_i[idx]))
        x_j = load_video(os.path.join(self.video_dir, self.fnames_j[idx]))

        # Sample frames to form clip from each "view"
        x_i = self._sample_frames(x_i)
        x_j = self._sample_frames(x_j)

        # Augment each view and obtain frame ordering label
        x_i, reordering_i = self._augment(x_i)
        x_j, reordering_j = self._augment(x_j)
        reordering_i = np.array(reordering_i)
        reordering_j = np.array(reordering_j)

        # Min-max normalize and swap axes for PyTorch
        x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min())
        x_j = (x_j - x_j.min()) / (x_j.max() - x_j.min())

        x_i = np.transpose(x_i, (3, 0, 1, 2))
        x_j = np.transpose(x_j, (3, 0, 1, 2))

        return (
            torch.from_numpy(x_i).float(),
            torch.from_numpy(x_j).float(),
            torch.from_numpy(reordering_i).long(),
            torch.from_numpy(reordering_j).long(),
        )


class EchoNetLVHDataset(EchoDataset):
    """
    Variant of EchoDataset for EchoNet-LVH, which has exactly one video per patient. Instead of pairing two different videos from the same study,
    each positive pair is formed from the *same* video: two clips are subsampled (from independent random start frames by default), resized,
    and independently augmented with the same light spatial augmentation + frame re-ordering as EchoDataset.

    Attributes
    ----------
    split: str
        Data split used to select videos (one of ["train", "val", "test"], matching the "split" column of MeasurementsList.csv)
    clip_len : int
        Number of frames to form video clips for training (clip length)
    sampling_rate : int
        Temporal "stride" when sampling frames to form clips
    frame_size : int
        Side length (pixels) that each frame is resized to
    same_clip : bool
        If True, both views use the same clip and differ only by augmentation; otherwise each view samples its own clip
    fnames : list[str]
        List of full paths to videos found on disk for this split
    temporal_orderings : list[tuple]
        List of all permutations of frame indices (target classes for frame re-ordering task)
    """

    def __init__(
        self,
        data_dir,
        split="train",
        clip_len=4,
        sampling_rate=1,
        frame_size=112,
        same_clip=False,
        n=None,
    ):
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.same_clip = same_clip

        label_df = pd.read_csv(os.path.join(data_dir, "MeasurementsList.csv"))
        names = np.sort(
            label_df.loc[label_df["split"] == self.split, "HashedFileName"].unique()
        )

        batch_dirs = sorted(
            d
            for d in os.listdir(data_dir)
            if d.startswith("Batch") and os.path.isdir(os.path.join(data_dir, d))
        )
        available = {}
        for batch_dir in batch_dirs:
            for fname in os.listdir(os.path.join(data_dir, batch_dir)):
                if fname.endswith(".avi"):
                    available.setdefault(
                        fname[:-4], os.path.join(data_dir, batch_dir, fname)
                    )

        self.fnames = [available[name] for name in names if name in available]
        print(
            f"{self.split}: found {len(self.fnames)} of {len(names)} videos ({len(names) - len(self.fnames)} missing on disk)"
        )

        if n is not None:
            self.fnames = self.fnames[:n]

        self.temporal_orderings = [
            _ for _ in itertools.permutations(np.arange(self.clip_len))
        ]

    def _resize(self, x):
        return np.stack(
            [
                cv2.resize(
                    frame,
                    (self.frame_size, self.frame_size),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in x
            ],
            axis=0,
        )

    @staticmethod
    def _normalize(x):
        x = x.astype(np.float32)
        return (x - x.min()) / max(x.max() - x.min(), 1e-8)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        x = load_video(self.fnames[idx])

        # Sample two clips from the same video (same clip if same_clip=True)
        x_i = self._resize(self._sample_frames(x))
        x_j = x_i.copy() if self.same_clip else self._resize(self._sample_frames(x))

        # Independently augment each view and obtain frame ordering label
        x_i, reordering_i = self._augment(x_i)
        x_j, reordering_j = self._augment(x_j)

        # Min-max normalize and swap axes for PyTorch
        x_i = np.transpose(self._normalize(x_i), (3, 0, 1, 2))
        x_j = np.transpose(self._normalize(x_j), (3, 0, 1, 2))

        return (
            torch.from_numpy(x_i).float(),
            torch.from_numpy(x_j).float(),
            torch.tensor(reordering_i).long(),
            torch.tensor(reordering_j).long(),
        )
