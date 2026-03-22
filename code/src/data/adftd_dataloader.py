import os
import numpy as np
import torch
import json
import random

from torch.utils.data import Dataset, DataLoader


def normalize_batch_ts(batch):
    mean_values = batch.mean(axis=1, keepdims=True)
    std_values = batch.std(axis=1, keepdims=True)
    std_values[std_values == 0] = 1.0
    return (batch - mean_values) / std_values


def collate_fn(data, max_len=None):
    batch_size = len(data)
    features, labels = zip(*data)

    lengths = [x.shape[0] for x in features]
    if max_len is None:
        max_len = max(lengths)

    x_out = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=features[0].dtype)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        x_out[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)
    masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    return x_out, targets, masks


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    if max_len is None:
        max_len = int(lengths.max().item())
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )

def get_id_list_adftd(args, data_list: np.ndarray, a=0.6, b=0.8):
    all_ids = list(data_list[:, 1])
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
    ad_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
    ftd_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])

    if args.cross_val in ["fixed", "mccv"]:
        rng = random.Random(42 if args.cross_val == "fixed" else args.seed)
        rng.shuffle(hc_list)
        rng.shuffle(ad_list)
        rng.shuffle(ftd_list)

        train_ids = (
            hc_list[: int(a * len(hc_list))]
            + ad_list[: int(a * len(ad_list))]
            + ftd_list[: int(a * len(ftd_list))]
        )
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
        )
        test_ids = (
            hc_list[int(b * len(hc_list)) :]
            + ad_list[int(b * len(ad_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
        )

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    if args.cross_val == "loso":
        if args.classify_choice == "ad_vs_hc":
            hc_ad_list = sorted(hc_list + ad_list)
            test_ids = [hc_ad_list[(args.seed - 41) % len(hc_ad_list)]]
            train_ids = [sid for sid in hc_ad_list if sid not in test_ids]
        else:
            all_ids = sorted(all_ids)
            test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
            train_ids = [sid for sid in all_ids if sid not in test_ids]

        rng = random.Random(args.seed)
        rng.shuffle(train_ids)
        val_ids = train_ids
        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    raise ValueError("Invalid cross_val. Please use fixed, mccv, or loso.")


class ADFTDRSDataset(Dataset):
    def __init__(self, args, root_path, flag):
        super().__init__()
        self.args = args
        self.no_normalize = args.no_normalize

        dataset_root = os.path.join(root_path, args.dataset_name)
        meta_path = os.path.join(dataset_root, "meta.json")
        x_path = os.path.join(dataset_root, "X.dat")
        y_path = os.path.join(dataset_root, "y.dat")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found at {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        n = int(meta["N"])
        t = int(meta["T"])
        c = int(meta["C"])

        self.X_mem = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, t, c))
        y_mem = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n, 3))

        subj_ids = y_mem[:, 1].astype(int)
        labels = y_mem[:, 0].astype(int)
        unique_sids = np.unique(subj_ids)
        subject_table = np.asarray([[int(labels[np.where(subj_ids == sid)[0][0]]), int(sid)] for sid in unique_sids])

        a, b = args.ratio_a, args.ratio_b
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_adftd(args, subject_table, a, b)

        if flag == "TRAIN":
            ids = self.train_ids
        elif flag == "VAL":
            ids = self.val_ids
        elif flag == "TEST":
            ids = self.test_ids
        else:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        ids = np.asarray(ids, dtype=int)
        mask = np.isin(subj_ids, ids)
        self.indices = np.where(mask)[0].astype(int)
        self.y = np.asarray(y_mem[self.indices])

        sampling_rate_list = list(map(int, args.sampling_rate_list.split(",")))
        sampling_mask = np.isin(self.y[:, 2], sampling_rate_list)
        if sampling_mask.sum() == 0:
            raise RuntimeError(
                f"No matching sampling rates. Found={np.unique(self.y[:, 2])}, target={sampling_rate_list}"
            )
        self.indices = self.indices[sampling_mask]
        self.y = self.y[sampling_mask]

        if args.classify_choice == "ad_vs_hc":
            label_mask = self.y[:, 0] < 2
            self.indices = self.indices[label_mask]
            self.y = self.y[label_mask]
        elif args.classify_choice == "ad_vs_nonad":
            self.y[:, 0] = np.where(self.y[:, 0] > 1, 0, self.y[:, 0])
        elif args.classify_choice == "hc_vs_abnormal":
            self.y[:, 0] = np.where(self.y[:, 0] > 1, 1, self.y[:, 0])
        elif args.classify_choice == "multi_class":
            pass
        else:
            raise ValueError(f"Unknown classify_choice={args.classify_choice}")

        self.max_seq_len = t
        self.enc_in = c
        self.num_class = int(np.unique(self.y[:, 0].astype(int)).shape[0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x_np = self.X_mem[real_idx]
        y_np = self.y[idx]

        if not self.no_normalize:
            x_np = normalize_batch_ts(x_np[np.newaxis, ...])[0]

        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
        return x, y


def build_loader(args, flag):
    ds = ADFTDRSDataset(args=args, root_path=args.root_path, flag=flag)
    is_train = flag == "TRAIN"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
    )
    return ds, loader
