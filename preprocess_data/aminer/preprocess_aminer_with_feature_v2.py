# downstream_build_features.py
import numpy as np
import pandas as pd

CSV_TRAIN = '../../processed_data/aminer/pp_aminer_half_year_train.csv'
CSV_VAL = '../../processed_data/aminer/pp_aminer_half_year_val.csv'
CSV_TEST = '../../processed_data/aminer/pp_aminer_half_year_test.csv'
CSV_ALL = '../../processed_data/aminer/pp_aminer_half_year_all.csv'

FEAT_TRAIN = '../../processed_data/aminer/pp_aminer_half_year_train_feature.npy'
FEAT_VAL = '../../processed_data/aminer/pp_aminer_half_year_val_feature.npy'
FEAT_TEST = '../../processed_data/aminer/pp_aminer_half_year_test_feature.npy'
FEAT_ALL = '../../processed_data/aminer/pp_aminer_half_year_all_feature.npy'

DAY = 60 * 60 * 24
front_half_year_seconds = DAY * (31 + 28 + 31 + 30 + 31 + 30)
behind_half_year_seconds = DAY * (31 + 31 + 30 + 31 + 30 + 31)
SEQ_L = 5
P_DIM = 7


def build_features_all(data_all: pd.DataFrame) -> np.ndarray:
    N = len(data_all)
    feats = np.zeros((N, SEQ_L, P_DIM), dtype=np.float32)

    # Align with original rows: use row indices for each item
    ts_all = data_all['ts'].to_numpy()
    u_all = data_all['u'].to_numpy()
    i_all = data_all['i'].to_numpy()
    label_all = data_all['label'].to_numpy()

    # Global time boundaries (avoid relying on sorting)
    start_time_all = int(data_all['ts'].min())
    end_time_all = int(data_all['ts'].max())

    # Group by item and reproduce half-year statistics
    groups = data_all.groupby('i', sort=False).groups  # dict: item_id -> Int64Index(row indices)
    for item_id, idxs_pd in groups.items():
        idxs = np.asarray(idxs_pd, dtype=np.int64)
        timestamps = ts_all[idxs]
        users = u_all[idxs]
        labels = label_all[idxs]

        # Sort within group by time while keeping mapping to original rows
        ord_ = np.argsort(timestamps, kind='mergesort')
        t_s = timestamps[ord_]
        u_s = users[ord_]
        y_s = labels[ord_]
        row_s = idxs[ord_]

        # Sliding alternating half-year windows
        half_year_count = 0
        start_time = start_time_all
        from numpy import unique, intersect1d

        # Historical unique user set
        before_users = np.array([], dtype=np.int64)

        # Feature sequence cache (most recent SEQ_L half-years)
        feats_seq = []

        while start_time + front_half_year_seconds <= end_time_all:
            end_time = start_time + (
                front_half_year_seconds if (half_year_count % 2 == 0)
                else behind_half_year_seconds
            )
            if half_year_count == 0:
                # (start, end] left-open right-closed
                start_time = start_time - 1

            # Indices within this half-year window
            mask = (t_s > start_time) & (t_s <= end_time)
            win_idxs = np.where(mask)[0]

            if win_idxs.size > 0:
                # Last interaction at the end of this half-year
                valid_pos = win_idxs[-1]

                # Compute 7 features
                current_popularity = int(win_idxs.size)
                current_users = u_s[win_idxs]
                unique_users, counts = unique(current_users, return_counts=True)
                current_num_users = int(unique_users.size)
                dup_mask = counts > 1
                current_num_duplicate_users = int(np.sum(dup_mask))
                current_num_duplicates = int(np.sum(counts[dup_mask] - 1))
                days_gap = float((end_time - t_s[valid_pos]) / DAY)

                if before_users.size > 0 and current_num_users > 0:
                    common_users = intersect1d(
                        unique_users, before_users, assume_unique=False
                    )
                    num_common_users = int(common_users.size)
                    if num_common_users > 0:
                        user_count_map = dict(
                            zip(unique_users.tolist(), counts.tolist())
                        )
                        total_occurrences = int(
                            sum(user_count_map[u_] for u_ in common_users)
                        )
                    else:
                        total_occurrences = 0
                else:
                    num_common_users = 0
                    total_occurrences = 0

                # Update historical user set
                before_users = np.union1d(before_users, unique_users)

                new_row = np.array([
                    current_popularity,
                    current_num_users,
                    current_num_duplicate_users,
                    current_num_duplicates,
                    days_gap,
                    num_common_users,
                    total_occurrences
                ], dtype=np.float32)
            else:
                # Empty half-year: all-zero feature
                new_row = np.zeros((P_DIM,), dtype=np.float32)

            # Maintain last SEQ_L half-year features
            feats_seq.append(new_row)
            if len(feats_seq) < SEQ_L:
                pad = [
                    np.zeros((P_DIM,), dtype=np.float32)
                    for _ in range(SEQ_L - len(feats_seq))
                ]
                feats_tail = np.stack(pad + feats_seq, axis=0)
            else:
                feats_tail = np.stack(feats_seq[-SEQ_L:], axis=0)

            # Write features back to the corresponding row if label != 0
            if win_idxs.size > 0:
                valid_pos = win_idxs[-1]
                if y_s[valid_pos] != 0:
                    row_id = row_s[valid_pos]
                    feats[row_id] = feats_tail

            start_time = end_time
            half_year_count += 1

    return feats


def save_splits_and_check(data_all: pd.DataFrame, features_all: np.ndarray):
    """
    Save train/val/test/all feature .npy files and perform checks:
      - Row count matches CSV
      - label == 0  <-> all-zero features
      - label != 0  <-> non-zero features
      - Valid edge count (label != 0) matches non-zero feature count
    Only prints check results
    """
    # Load splits (use upstream saved CSVs)
    df_train = pd.read_csv(CSV_TRAIN)
    df_val = pd.read_csv(CSV_VAL)
    df_test = pd.read_csv(CSV_TEST)
    df_all = data_all

    # Use CSV time boundaries to construct masks
    ts_all = data_all['ts'].to_numpy()
    t_train_max = df_train['ts'].max() if len(df_train) > 0 else -np.inf
    t_val_max = df_val['ts'].max() if len(df_val) > 0 else -np.inf

    mask_train = (ts_all <= t_train_max)
    mask_val = (ts_all > t_train_max) & (ts_all <= t_val_max)
    mask_test = (ts_all > t_val_max)

    # Save .npy files
    np.save(FEAT_TRAIN, features_all[mask_train])
    np.save(FEAT_VAL, features_all[mask_val])
    np.save(FEAT_TEST, features_all[mask_test])
    np.save(FEAT_ALL, features_all)

    # ---- Check function (print results only) ----
    def check(split_name, df_split, feats_split, show_k: int = 10):
        ok = True
        # 1) Row count consistency
        if len(df_split) != len(feats_split):
            print(
                f"[{split_name}] Row count mismatch: "
                f"csv={len(df_split)} vs npy={len(feats_split)}"
            )
            ok = False

        labels = df_split['label'].to_numpy()
        flat = (
            feats_split.reshape(feats_split.shape[0], -1)
            if len(feats_split) > 0
            else np.zeros((0, SEQ_L * P_DIM))
        )
        all_zero_mask = (
            (flat == 0).all(axis=1) if len(flat) > 0
            else np.array([], dtype=bool)
        )

        # 2) label == 0 <-> all-zero features
        nz_label = (labels != 0)
        z_label = ~nz_label
        nz_feat = ~all_zero_mask
        z_feat = all_zero_mask

        a = np.sum(z_label)
        b = np.sum(z_feat)
        c = np.sum(nz_label)
        d = np.sum(nz_feat)

        if a != b:
            print(
                f"[{split_name}] label==0 count={a} "
                f"!= all-zero feature count={b}"
            )
            ok = False
        if c != d:
            print(
                f"[{split_name}] label!=0 count={c} "
                f"!= non-zero feature count={d}"
            )
            ok = False

        # 3) Valid edge count consistency
        print(
            f"[{split_name}] OK={ok} | rows={len(df_split)} | "
            f"label==0={a} <-> zero_feats={b} | "
            f"label!=0={c} <-> nonzero_feats={d}"
        )

        # Per-row consistency (by row index)
        per_row_ok = (z_label == z_feat)
        bad_idx = np.where(~per_row_ok)[0]
        if bad_idx.size > 0:
            print(
                f"[{split_name}] Per-row mismatch {bad_idx.size} rows "
                f"(showing first {min(show_k, bad_idx.size)}): "
                f"{bad_idx[:show_k].tolist()}"
            )
            ok = False

        return ok, c, d

    ok1, c1, d1 = check("train", df_train, np.load(FEAT_TRAIN, allow_pickle=False))
    ok2, c2, d2 = check("val", df_val, np.load(FEAT_VAL, allow_pickle=False))
    ok3, c3, d3 = check("test", df_test, np.load(FEAT_TEST, allow_pickle=False))
    ok4, c4, d4 = check("all", df_all, np.load(FEAT_ALL, allow_pickle=False))

    # Print valid edge counts
    print(
        f"[valid edge num] train={c1}(labels) / {d1}(feats) | "
        f"val={c2}/{d2} | test={c3}/{d3} | all={c4}/{d4}"
    )


def main():
    # Load ALL (other CSVs are only for checking and masks)
    data_all = pd.read_csv(CSV_ALL)
    # Add label column if missing
    if 'label' not in data_all.columns:
        data_all['label'] = 0

    # Build features aligned with all rows
    features_all = build_features_all(data_all)

    # Save splits and run checks
    save_splits_and_check(data_all, features_all)


if __name__ == "__main__":
    main()
