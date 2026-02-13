import numpy as np
import pandas as pd

CSV_TRAIN = '../../processed_data/aminer/pp_aminer_half_year_train_sp.csv'
CSV_VAL = '../../processed_data/aminer/pp_aminer_half_year_val_sp.csv'
CSV_TEST = '../../processed_data/aminer/pp_aminer_half_year_test_sp.csv'

FEAT_TRAIN = '../../processed_data/aminer/pp_aminer_half_year_train_sp.npy'
FEAT_VAL = '../../processed_data/aminer/pp_aminer_half_year_val_sp.npy'
FEAT_TEST = '../../processed_data/aminer/pp_aminer_half_year_test_sp.npy'
FEAT_ALL = '../../processed_data/aminer/pp_aminer_half_year_all_sp.npy'

OLD_FEAT_TRAIN = '../../processed_data/aminer/pp_aminer_half_year_train_feature.npy'
OLD_FEAT_VAL   = '../../processed_data/aminer/pp_aminer_half_year_val_feature.npy'
OLD_FEAT_TEST  = '../../processed_data/aminer/pp_aminer_half_year_test_feature.npy'

DAY = 60 * 60 * 24
front_half_year_seconds = DAY * (31 + 28 + 31 + 30 + 31 + 30)  # Jan–Jun
behind_half_year_seconds = DAY * (31 + 31 + 30 + 31 + 30 + 31)  # Jul–Dec
SEQ_L = 5
P_DIM = 4  # ps in {0,1,2,3} counts


def build_features_all(data_all: pd.DataFrame) -> np.ndarray:
    """
    Generate aligned (5,4) features for each row in data_all:
      - Default is all -1 (indicating "no prediction / no write")
      - If the row is a prediction point (last interaction of the half-year and label!=0),
        write the most recent 5 half-year statistics.
    Return: features_all, shape=(len(data_all), 5, 4)
    """
    N = len(data_all)
    feats = -np.ones((N, SEQ_L, P_DIM), dtype=np.float32)

    ts_all = data_all['ts'].to_numpy()
    u_all = data_all['u'].to_numpy()
    i_all = data_all['i'].to_numpy()
    label_all = data_all['label'].to_numpy()

    if 'sp' not in data_all.columns:
        raise ValueError("Column `ps` is missing in data_all")
    ps_all = data_all['sp'].to_numpy()

    start_time_all = int(data_all['ts'].min())
    end_time_all = int(data_all['ts'].max())

    groups = data_all.groupby('i', sort=False).groups
    for item_id, idxs_pd in groups.items():
        idxs = np.asarray(idxs_pd, dtype=np.int64)

        timestamps = ts_all[idxs]
        labels = label_all[idxs]
        ps_vals = ps_all[idxs]

        ord_ = np.argsort(timestamps, kind='mergesort')
        t_s = timestamps[ord_]
        y_s = labels[ord_]
        ps_s = ps_vals[ord_]
        row_s = idxs[ord_]

        half_year_count = 0
        start_time = start_time_all

        feats_seq = []  # each element is (4,) ps 0/1/2/3 counts

        while start_time + front_half_year_seconds <= end_time_all:
            end_time = start_time + (front_half_year_seconds if (half_year_count % 2 == 0)
                                     else behind_half_year_seconds)
            if half_year_count == 0:
                start_time = start_time - 1  # (start, end]

            mask = (t_s > start_time) & (t_s <= end_time)
            win_idxs = np.where(mask)[0]

            if win_idxs.size > 0:
                win_ps = ps_s[win_idxs].astype(np.int64)

                # If ps contains values outside 0..3, bincount will still count them;
                # only the first 4 dimensions are used.
                cnt = np.bincount(win_ps, minlength=4)[:4].astype(np.float32)
                new_row = cnt
            else:
                new_row = np.zeros((P_DIM,), dtype=np.float32)

            feats_seq.append(new_row)

            # Maintain the most recent SEQ_L half-year features (left-pad with zeros if insufficient)
            if len(feats_seq) < SEQ_L:
                pad = [np.zeros((P_DIM,), dtype=np.float32) for _ in range(SEQ_L - len(feats_seq))]
                feats_tail = np.stack(pad + feats_seq, axis=0)  # (5,4)
            else:
                feats_tail = np.stack(feats_seq[-SEQ_L:], axis=0)  # (5,4)

            # Only write at the last interaction of this half-year and label!=0
            if win_idxs.size > 0:
                valid_pos = win_idxs[-1]
                if y_s[valid_pos] != 0:
                    row_id = row_s[valid_pos]
                    feats[row_id] = feats_tail

            start_time = end_time
            half_year_count += 1

    return feats


def save_splits_and_check(data_all: pd.DataFrame, features_all: np.ndarray):
    df_train = pd.read_csv(CSV_TRAIN)
    df_val = pd.read_csv(CSV_VAL)
    df_test = pd.read_csv(CSV_TEST)
    df_all = data_all

    ts_all = data_all['ts'].to_numpy()
    t_train_max = df_train['ts'].max() if len(df_train) > 0 else -np.inf
    t_val_max = df_val['ts'].max() if len(df_val) > 0 else -np.inf

    mask_train = (ts_all <= t_train_max)
    mask_val = (ts_all > t_train_max) & (ts_all <= t_val_max)
    mask_test = (ts_all > t_val_max)

    # Save new features
    np.save(FEAT_TRAIN, features_all[mask_train])
    np.save(FEAT_VAL, features_all[mask_val])
    np.save(FEAT_TEST, features_all[mask_test])
    np.save(FEAT_ALL, features_all)

    def check(split_name, df_split, feats_split, show_k: int = 10):
        ok = True
        if len(df_split) != len(feats_split):
            print(f"[{split_name}] Row count mismatch: csv={len(df_split)} vs npy={len(feats_split)}")
            ok = False

        labels = df_split['label'].to_numpy()
        flat = feats_split.reshape(feats_split.shape[0], -1) if len(feats_split) > 0 else np.zeros((0, SEQ_L * P_DIM))

        # Locations not predicted should be all -1
        all_minus1_mask = (flat == -1).all(axis=1) if len(flat) > 0 else np.array([], dtype=bool)

        nz_label = (labels != 0)
        z_label = ~nz_label
        nz_feat = ~all_minus1_mask
        z_feat = all_minus1_mask

        a = int(np.sum(z_label))
        b = int(np.sum(z_feat))
        c = int(np.sum(nz_label))
        d = int(np.sum(nz_feat))

        if a != b:
            print(f"[{split_name}] label==0 count={a} does not match all-(-1) feature count={b}")
            ok = False
        if c != d:
            print(f"[{split_name}] label!=0 count={c} does not match written feature count={d}")
            ok = False

        print(f"[{split_name}] OK={ok} | rows={len(df_split)} | "
              f"label==0={a} ↔ minus1_feats={b} | "
              f"label!=0={c} ↔ written_feats={d}")

        per_row_ok = (z_label == z_feat)
        bad_idx = np.where(~per_row_ok)[0]
        if bad_idx.size > 0:
            print(f"[{split_name}] Row-level mismatch {bad_idx.size} entries "
                  f"(showing first {min(show_k, bad_idx.size)} indices): "
                  f"{bad_idx[:show_k].tolist()}")
            ok = False

        return ok, c, d

    # ===== Basic consistency check (label vs -1 padding) =====
    ok1, c1, d1 = check("train", df_train, np.load(FEAT_TRAIN, allow_pickle=False))
    ok2, c2, d2 = check("val", df_val, np.load(FEAT_VAL, allow_pickle=False))
    ok3, c3, d3 = check("test", df_test, np.load(FEAT_TEST, allow_pickle=False))
    ok4, c4, d4 = check("all", df_all, np.load(FEAT_ALL, allow_pickle=False))

    print(f"[valid edge num] train={c1}(labels) / {d1}(feats) | "
          f"val={c2}/{d2} | test={c3}/{d3} | all={c4}/{d4}")

    # ===== Additional comparison: check sum(ps) == old_popularity for label!=0 rows =====
    def compare_sum_with_old(split_name, csv_path, new_feat_path, old_feat_path, show_k: int = 10):
        df = pd.read_csv(csv_path)
        labels = df['label'].to_numpy()

        new_feats = np.load(new_feat_path, allow_pickle=False)
        old_feats = np.load(old_feat_path, allow_pickle=False)

        if len(labels) != new_feats.shape[0] or len(labels) != old_feats.shape[0]:
            print(f"[{split_name}][compare] Row count mismatch: "
                  f"csv={len(labels)} new={new_feats.shape[0]} old={old_feats.shape[0]}")
            return

        check_mask = (labels != 0)
        n_check = int(check_mask.sum())

        new_sum = new_feats.sum(axis=2)
        old_pop = old_feats[:, :, 0].astype(np.float32)

        diff = new_sum - old_pop
        bad = np.where(check_mask & (np.abs(diff).max(axis=1) != 0))[0]

        ok = (bad.size == 0)
        print(f"[{split_name}][compare sum==old_pop] OK={ok} | "
              f"checked_rows={n_check} | bad_rows={bad.size}")

        if bad.size > 0:
            print(f"[{split_name}][compare] showing first {min(show_k, bad.size)} mismatches:")
            for rid in bad[:min(show_k, bad.size)]:
                bad_windows = np.where(diff[rid] != 0)[0].tolist()
                print(f"  - row={int(rid)} bad_windows={bad_windows} label={labels[rid]}")
                print(f"    old_pop={old_pop[rid].tolist()}")
                print(f"    new_sum={new_sum[rid].tolist()}")
                print(f"    new_feats={new_feats[rid].tolist()}")

    compare_sum_with_old("train", CSV_TRAIN, FEAT_TRAIN, OLD_FEAT_TRAIN)
    compare_sum_with_old("val",   CSV_VAL,   FEAT_VAL,   OLD_FEAT_VAL)
    compare_sum_with_old("test",  CSV_TEST,  FEAT_TEST,  OLD_FEAT_TEST)


def main():
    # Read ALL (others are only used for checking / determining masks)
    df_train = pd.read_csv(CSV_TRAIN)
    df_val = pd.read_csv(CSV_VAL)
    df_test = pd.read_csv(CSV_TEST)
    data_all = pd.concat(
        [df_train, df_val, df_test],
        axis=0,
        ignore_index=True
    )

    # If label column does not exist (theoretically unlikely), add it
    if 'label' not in data_all.columns:
        data_all['label'] = 0

    # Generate aligned full features
    features_all = build_features_all(data_all)

    # Save splits and perform checks (only output checking results)
    save_splits_and_check(data_all, features_all)


if __name__ == "__main__":
    main()
