import argparse

import os
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import optim

from model.eegconformer import EEGConformerModel
from data.adftd_dataloader import ADFTDRSDataset, build_loader

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone supervised EEGConformer training on ADFTD-RS"
    )

    parser.add_argument("--root_path", type=str, default="./dataset/L400")
    parser.add_argument("--dataset_name", type=str, default="ADFTD-RS")
    parser.add_argument("--classify_choice", type=str, default="multi_class")
    parser.add_argument("--cross_val", type=str, default="mccv", choices=["fixed", "mccv", "loso"])
    parser.add_argument("--ratio_a", type=float, default=0.8)
    parser.add_argument("--ratio_b", type=float, default=0.9)
    parser.add_argument("--sampling_rate_list", type=str, default="200")
    parser.add_argument("--no_normalize", action="store_true", default=False)

    parser.add_argument("--e_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--use_subject_vote", action="store_true", default=False)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/standalone_eegconformer_adftd_rs")

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    return float(np.mean(specificities))


def calculate_subject_level_metrics(predictions, true_labels, subject_ids, num_classes):
    unique_subjects = np.unique(subject_ids)
    subject_predictions = []
    subject_trues = []

    for subject in unique_subjects:
        idx = np.where(subject_ids == subject)[0]
        subj_preds = predictions[idx]
        subj_true = true_labels[idx][0]
        majority_label = Counter(subj_preds).most_common(1)[0][0]
        subject_predictions.append(majority_label)
        subject_trues.append(subj_true)

    subject_predictions = np.asarray(subject_predictions)
    subject_trues = np.asarray(subject_trues)

    metrics = {"Accuracy": accuracy_score(subject_trues, subject_predictions)}
    unique_labels = np.unique(subject_trues)
    if len(unique_labels) < 2:
        metrics.update(
            {
                "Precision": -1,
                "Recall": -1,
                "Specificity": -1,
                "F1": -1,
                "AUROC": -1,
                "AUPRC": -1,
            }
        )
        return metrics

    try:
        true_onehot = np.eye(num_classes)[subject_trues.astype(int)]
        pred_onehot = np.eye(num_classes)[subject_predictions.astype(int)]
        metrics["Precision"] = precision_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["Specificity"] = multiclass_specificity(subject_trues, subject_predictions)
        metrics["F1"] = f1_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["AUROC"] = roc_auc_score(true_onehot, pred_onehot, multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(true_onehot, pred_onehot, average="macro")
    except ValueError:
        metrics["Precision"] = precision_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["Specificity"] = multiclass_specificity(subject_trues, subject_predictions)
        metrics["F1"] = f1_score(subject_trues, subject_predictions, average="macro", zero_division=0)
        metrics["AUROC"] = -1
        metrics["AUPRC"] = -1

    return metrics


def compute_metrics(pred_logits, true_labels):
    probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    y_true = true_labels.cpu().numpy()

    metrics = {"Accuracy": accuracy_score(y_true, preds)}
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        metrics.update(
            {
                "Precision": -1,
                "Recall": -1,
                "Specificity": -1,
                "F1": -1,
                "AUROC": -1,
                "AUPRC": -1,
            }
        )
        return metrics, preds

    onehot = torch.nn.functional.one_hot(
        true_labels.to(torch.long), num_classes=pred_logits.shape[1]
    ).cpu().numpy()

    metrics["Precision"] = precision_score(y_true, preds, average="macro", zero_division=0)
    metrics["Recall"] = recall_score(y_true, preds, average="macro", zero_division=0)
    metrics["Specificity"] = multiclass_specificity(y_true, preds)
    metrics["F1"] = f1_score(y_true, preds, average="macro", zero_division=0)

    try:
        metrics["AUROC"] = roc_auc_score(onehot, probs, multi_class="ovr")
        metrics["AUPRC"] = average_precision_score(onehot, probs, average="macro")
    except ValueError:
        metrics["AUROC"] = -1
        metrics["AUPRC"] = -1

    return metrics, preds


def evaluate(model, loader, device, criterion, use_subject_vote=False):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    all_sub_ids = []

    with torch.no_grad():
        for batch_x, label_id, padding_mask in loader:
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)

            labels = label_id[:, 0].long().to(device)
            sub_ids = label_id[:, 1].long().cpu()
            fs = label_id[:, 2]

            logits = model(batch_x, padding_mask, None, None, fs, None)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sub_ids.append(sub_ids)

    mean_loss = float(np.mean(losses)) if losses else 0.0
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sub_ids = torch.cat(all_sub_ids, dim=0).numpy()

    sample_metrics, preds = compute_metrics(logits, labels)
    subject_metrics = None
    if use_subject_vote:
        subject_metrics = calculate_subject_level_metrics(
            predictions=preds,
            true_labels=labels.numpy(),
            subject_ids=sub_ids,
            num_classes=logits.shape[1],
        )

    return mean_loss, sample_metrics, subject_metrics


def fmt_metrics(metrics):
    return ", ".join([f"{k}: {v:.4f}" if v != -1 else f"{k}: -1" for k, v in metrics.items()])


def main():
    args = parse_args()
    seed_everything(args.seed)

    args.method = "EEGConformer"
    args.task_name = "supervised"
    args.model = "EEGConformer"
    args.data = "ADFTD-RS"
    args.training_datasets = args.dataset_name
    args.testing_datasets = args.dataset_name
    args.output_attention = False

    tmp_train = ADFTDRSDataset(args=args, root_path=args.root_path, flag="TRAIN")
    args.seq_len = int(tmp_train.max_seq_len)
    args.enc_in = int(tmp_train.enc_in)
    args.num_class = int(tmp_train.num_class)

    train_ds, train_loader = build_loader(args, "TRAIN")
    val_ds, val_loader = build_loader(args, "VAL")
    test_ds, test_loader = build_loader(args, "TEST")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model_cfg = SimpleNamespace(**vars(args))
    model = EEGConformerModel(model_cfg).float().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best.pth")

    print("Args:", args)
    print(
        f"Data stats | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"seq_len={args.seq_len} enc_in={args.enc_in} num_class={args.num_class}"
    )

    best_val_f1 = -1.0
    wait = 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        batch_losses = []

        for batch_x, label_id, padding_mask in train_loader:
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            labels = label_id[:, 0].long().to(device)
            fs = label_id[:, 2]

            optimizer.zero_grad()
            logits = model(batch_x, padding_mask, None, None, fs, None)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, val_sample_metrics, val_subject_metrics = evaluate(
            model, val_loader, device, criterion, use_subject_vote=args.use_subject_vote
        )
        test_loss, test_sample_metrics, test_subject_metrics = evaluate(
            model, test_loader, device, criterion, use_subject_vote=args.use_subject_vote
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} test_loss={test_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6e}"
        )
        print("Sample metrics | val:", fmt_metrics(val_sample_metrics))
        print("Sample metrics | test:", fmt_metrics(test_sample_metrics))
        if args.use_subject_vote:
            print("Subject metrics | val:", fmt_metrics(val_subject_metrics))
            print("Subject metrics | test:", fmt_metrics(test_subject_metrics))

        current_val_f1 = val_sample_metrics.get("F1", -1)
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path} (val F1={best_val_f1:.4f})")
        else:
            wait += 1
            print(f"No validation F1 improvement for {wait} epoch(s).")

        if wait >= args.patience:
            print("Early stopping triggered.")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, val_sample_metrics, val_subject_metrics = evaluate(
        model, val_loader, device, criterion, use_subject_vote=args.use_subject_vote
    )
    _, test_sample_metrics, test_subject_metrics = evaluate(
        model, test_loader, device, criterion, use_subject_vote=args.use_subject_vote
    )

    print("Final (best-checkpoint) sample metrics | val:", fmt_metrics(val_sample_metrics))
    print("Final (best-checkpoint) sample metrics | test:", fmt_metrics(test_sample_metrics))
    if args.use_subject_vote:
        print("Final (best-checkpoint) subject metrics | val:", fmt_metrics(val_subject_metrics))
        print("Final (best-checkpoint) subject metrics | test:", fmt_metrics(test_subject_metrics))


if __name__ == "__main__":
    main()