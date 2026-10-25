import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
import tqdm
import matplotlib.pyplot as plt

from copy import deepcopy

from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer


def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir):
    """Train model to classify LVH for one epoch.

    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    model.train()

    pbar = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}"
    )
    running_loss = 0.0
    y_true, y_hat = [], []
    acc_nums = []
    plax_probs = []
    for i, batch in pbar:
        x, y = batch["x"].to(device), batch["y"].to(device)
        acc_num = batch["acc_num"]

        # Forward pass
        out = model(x)

        # Compute loss
        loss = loss_fxn(out, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Keep running sum of batch losses
        running_loss += loss.item()

        y_true.append(y.detach().cpu().numpy())
        y_hat.append(out.sigmoid().detach().cpu().numpy())
        acc_nums.append(acc_num)

        del out
        del loss

        pbar.set_postfix({"loss": running_loss / (i + 1)})  # , 'b_acc': balanced_acc})

    video_label_df = pd.DataFrame(
        {
            "y_true": np.concatenate(y_true).ravel(),
            "y_hat": np.concatenate(y_hat).ravel(),
            "acc_num": np.concatenate(acc_nums).ravel(),
        }
    )

    y_true = video_label_df["y_true"]
    y_hat = video_label_df["y_hat"]

    # AUROC
    auroc = metrics.roc_auc_score(y_true, y_hat)
    pr, re, _ = metrics.precision_recall_curve(y_true, y_hat)
    aupr = metrics.auc(re, pr)

    # Other metrics
    acc = metrics.accuracy_score(y_true, y_hat.round())
    b_acc = metrics.balanced_accuracy_score(y_true, y_hat.round())
    mcc = metrics.matthews_corrcoef(y_true, y_hat.round())
    pr, re, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_hat.round(), average="binary"
    )

    out_str = f"[VIDEO LEVEL]   AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}"
    print(out_str)

    # Append metrics to history df
    current_metrics = pd.DataFrame(
        [
            [
                epoch,
                "train",
                running_loss / (i + 1),
                auroc,
                aupr,
                acc,
                b_acc,
                mcc,
                pr,
                re,
                f1,
            ]
        ],
        columns=history.columns,
    )
    current_metrics.to_csv(
        os.path.join(model_dir, "history.csv"), mode="a", header=False, index=False
    )

    return pd.concat([history, current_metrics], ignore_index=True)


def validate(
    model,
    device,
    loss_fxn,
    optimizer,
    data_loader,
    history,
    epoch,
    model_dir,
    early_stopping_dict,
    best_model_wts,
):
    """Evaluate LVH model on validation set, saving weights if val loss has improved.

    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        early_stopping_dict: dict
            Dictionary with keys "best_loss" and "epochs_no_improve" for early stopping and model checkpointing
        best_model_wts: dict
            Dictionary containing PyTorch model weights as returned by `model.state_dict()`
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
        early_stopping_dict: dict
            Updated dictionary with information for early stopping and model checkpointing
        best_model_wts: dict
            Updated dictionary containing best PyTorch model weights
    """
    model.eval()

    pbar = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=f"[VAL] Epoch {epoch}"
    )
    running_loss = 0.0
    y_true, y_hat = [], []
    acc_nums = []
    plax_probs = []
    with torch.no_grad():
        for i, batch in pbar:
            x, y = batch["x"].to(device), batch["y"].to(device)
            acc_num = batch["acc_num"]

            # Forward pass
            out = torch.stack(
                [model(x[:, :, clip, :, :, :]) for clip in range(x.shape[2])], dim=0
            )

            # Get softmax probabilities and video-wise average over clip dimension
            out = out.sigmoid().mean(dim=0)

            # Compute loss
            loss = loss_fxn(out, y)

            # Keep running sum of batch losses
            running_loss += loss.item()

            y_true.append(y.detach().cpu().numpy())
            y_hat.append(out.detach().cpu().numpy())
            acc_nums.append(acc_num)

            del out
            del loss

            pbar.set_postfix(
                {"loss": running_loss / (i + 1)}
            )  # , 'b_acc': balanced_acc})

    video_label_df = pd.DataFrame(
        {
            "y_true": np.concatenate(y_true).ravel(),
            "y_hat": np.concatenate(y_hat).ravel(),
            "acc_num": np.concatenate(acc_nums).ravel(),
        }
    )

    y_true = video_label_df["y_true"]
    y_hat = video_label_df["y_hat"]

    # AUROC
    auroc = metrics.roc_auc_score(y_true, y_hat)
    pr, re, _ = metrics.precision_recall_curve(y_true, y_hat)
    aupr = metrics.auc(re, pr)

    # Other metrics
    acc = metrics.accuracy_score(y_true, y_hat.round())
    b_acc = metrics.balanced_accuracy_score(y_true, y_hat.round())
    mcc = metrics.matthews_corrcoef(y_true, y_hat.round())
    pr, re, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_hat.round(), average="binary"
    )

    out_str = f"[VIDEO LEVEL]   AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}"
    print(out_str)

    current_metrics = pd.DataFrame(
        [
            [
                epoch,
                "val",
                running_loss / (i + 1),
                auroc,
                aupr,
                acc,
                b_acc,
                mcc,
                pr,
                re,
                f1,
            ]
        ],
        columns=history.columns,
    )
    current_metrics.to_csv(
        os.path.join(model_dir, "history.csv"), mode="a", header=False, index=False
    )

    # Early stopping: save model weights only when val AUC has improved
    if auroc > early_stopping_dict["best_auroc"]:
        print(
            f"EARLY STOPPING: AUROC has improved from {round(early_stopping_dict['best_auroc'], 3)} to {round(auroc, 3)}! Saving weights."
        )
        early_stopping_dict["epochs_no_improve"] = 0
        early_stopping_dict["best_auroc"] = auroc
        best_model_wts = (
            deepcopy(model.state_dict())
            if not isinstance(model, torch.nn.DataParallel)
            else deepcopy(model.module.state_dict())
        )
        torch.save(
            {"weights": best_model_wts, "optimizer": optimizer.state_dict()},
            os.path.join(model_dir, f"chkpt_epoch-{epoch}.pt"),
        )
    else:
        print(
            f"EARLY STOPPING: AUROC has not improved from {round(early_stopping_dict['best_auroc'], 3)}"
        )
        early_stopping_dict["epochs_no_improve"] += 1

    return (
        pd.concat([history, current_metrics], ignore_index=True),
        early_stopping_dict,
        best_model_wts,
    )


def evaluate(
    model, device, loss_fxn, data_loader, split, classes, history, model_dir, weights
):
    """Evaluate LVH model on test set and save performance metrics (text file summary, plots, etc.)

    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        data_loader : PyTorch data loader
        split : str
            Data split used for evaluation (one of ["test", "ext_test"])
        classes : list[str]
            List of output class names
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights: dict
            Dictionary containing PyTorch model weights as returned by `model.state_dict()`
    Returns
    -------
    """
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights)
    else:
        model.load_state_dict(weights)
    model.eval()

    pbar = tqdm.tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"[{split.upper()}] EVALUATION",
    )
    running_loss = 0.0
    y_true, y_hat = [], []
    acc_nums = []
    with torch.no_grad():
        for i, batch in pbar:
            x, y = batch["x"].to(device), batch["y"].to(device)
            acc_num = batch["acc_num"]

            # Forward pass
            out = torch.stack(
                [model(x[:, :, clip, :, :, :]) for clip in range(x.shape[2])], dim=0
            )

            # Get softmax probabilities and video-wise average over clip dimension
            out = out.sigmoid().mean(dim=0)

            # Compute loss
            loss = loss_fxn(out, y)

            # Keep running sum of batch losses
            running_loss += loss.item()

            y_true.append(y.detach().cpu().numpy())
            y_hat.append(out.detach().cpu().numpy())
            acc_nums.append(acc_num)

            pbar.set_postfix(
                {"loss": running_loss / (i + 1)}
            )  # , 'b_acc': balanced_acc})

    print(split)
    video_label_df = pd.DataFrame(
        {
            "y_true": np.concatenate(y_true).ravel(),
            "y_hat": np.concatenate(y_hat).ravel(),
            "acc_num": np.concatenate(acc_nums).ravel(),
        }
    )

    print(video_label_df)

    y_true = video_label_df["y_true"]
    y_hat = video_label_df["y_hat"]

    # AUROC
    auroc = metrics.roc_auc_score(y_true, y_hat)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_hat)
    pr, re, _ = metrics.precision_recall_curve(y_true, y_hat)
    aupr = metrics.auc(re, pr)

    # Other metrics
    acc = metrics.accuracy_score(y_true, y_hat.round())
    b_acc = metrics.balanced_accuracy_score(y_true, y_hat.round())
    mcc = metrics.matthews_corrcoef(y_true, y_hat.round())
    pr, re, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_hat.round(), average="binary"
    )
    cls_report = metrics.classification_report(
        y_true, y_hat.round(), target_names=classes, digits=3
    )

    video_out_str = f"[VIDEO LEVEL]   AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}"
    video_out_str += f"\n{cls_report}"
    print(video_out_str)

    # SAVE OUTPUT
    video_label_df.to_csv(
        os.path.join(model_dir, f"{split}_video_preds.csv"), index=False
    )  # video-level predictions

    # Plot loss learing curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        history.loc[history["phase"] == "train", "epoch"],
        history.loc[history["phase"] == "train", "loss"],
        label="train",
    )
    ax.plot(
        history.loc[history["phase"] == "val", "epoch"],
        history.loc[history["phase"] == "val", "loss"],
        label="val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(
        os.path.join(model_dir, "loss_history.png"), dpi=300, bbox_inches="tight"
    )

    # Plot AUROC learning curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        history.loc[history["phase"] == "train", "epoch"],
        history.loc[history["phase"] == "train", "auroc"],
        label="train",
    )
    ax.plot(
        history.loc[history["phase"] == "val", "epoch"],
        history.loc[history["phase"] == "val", "auroc"],
        label="val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend()
    fig.savefig(
        os.path.join(model_dir, "auroc_history.png"), dpi=300, bbox_inches="tight"
    )

    # Plot AUPR (AP) learning curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        history.loc[history["phase"] == "train", "epoch"],
        history.loc[history["phase"] == "train", "aupr"],
        label="train",
    )
    ax.plot(
        history.loc[history["phase"] == "val", "epoch"],
        history.loc[history["phase"] == "val", "aupr"],
        label="val",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPR")
    ax.legend()
    fig.savefig(
        os.path.join(model_dir, "aupr_history.png"), dpi=300, bbox_inches="tight"
    )

    # Plot ROC
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"LVH (AUC: {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 - Specificity", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    fig.savefig(
        os.path.join(model_dir, f"{split}_roc.png"), dpi=300, bbox_inches="tight"
    )


def load_video(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    capture = cv2.VideoCapture(fpath)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for i in range(frame_count):
        ret, frame = capture.read()

        v[i] = frame

    return v


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
