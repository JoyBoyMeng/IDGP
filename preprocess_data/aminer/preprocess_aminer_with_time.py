import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, Deque, List, Tuple

def _get_global_ts_range(paths: List[str], ts_col: str) -> Tuple[float, float]:
    gmin = None
    gmax = None
    for p in paths:
        df = pd.read_csv(p, usecols=[ts_col])
        ts = df[ts_col].to_numpy(dtype=np.float64)
        cur_min = float(np.min(ts))
        cur_max = float(np.max(ts))
        gmin = cur_min if gmin is None else min(gmin, cur_min)
        gmax = cur_max if gmax is None else max(gmax, cur_max)
    return gmin, gmax

def build_item_history_dt_npys(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_train_npy: str,
    out_val_npy: str,
    out_test_npy: str,
    out_all_npy: str,
    N: int = 100,
    item_col: str = "i",
    ts_col: str = "ts",
    label_col: str = "label",
    recent_first: bool = True,   # True: most recent history first; False: oldest first
    out_dtype = np.float32,
    verify_show_k: int = 10,     # maximum number of error rows to display during verification
):
    paths = [train_csv, val_csv, test_csv]

    # Global time span of the full graph (used for padding)
    gmin, gmax = _get_global_ts_range(paths, ts_col)
    print("global ts range:", gmin, gmax)
    max_span = float(gmax - gmin)

    # Shared history across splits
    hist: Dict[object, Deque[float]] = defaultdict(lambda: deque(maxlen=N))

    def process_one(csv_path: str, out_path: str):
        df = pd.read_csv(csv_path, usecols=[item_col, ts_col, label_col])

        items = df[item_col].to_numpy()
        ts = df[ts_col].to_numpy(dtype=np.float64)
        labels = df[label_col].to_numpy()

        M = len(df)

        # Initialize with -1 (rows with label==0 remain -1)
        out = np.full((M, N), fill_value=-1, dtype=out_dtype)

        for idx in range(M):
            i = items[idx]
            t = float(ts[idx])
            lbl = labels[idx]

            # Only output time sequence when label!=0; label==0 remains -1
            if lbl != 0:
                h = hist[i]  # history before current interaction

                if len(h) > 0:
                    h_arr = np.fromiter(h, dtype=np.float64, count=len(h))  # oldest -> newest
                    if recent_first:
                        h_arr = h_arr[::-1]  # newest -> oldest
                    deltas = t - h_arr
                    L = deltas.shape[0]

                    out[idx, :L] = deltas.astype(out_dtype, copy=False)
                    if L < N:
                        out[idx, L:] = max_span
                else:
                    # No history but label!=0: entire row is padding max_span
                    out[idx, :] = max_span

            # Update history regardless of label value
            hist[i].append(t)

        np.save(out_path, out)
        print(f"Saved: {out_path} | shape={out.shape} dtype={out.dtype} pad(max_span)={max_span}")

    def final_verify():
        """
        Final verification (executed after all splits are processed):

        - Rows with label==0: npy must be entirely -1
        - Rows with label!=0: npy must not contain -1
        - Output number of valid rows (rows not entirely -1) for each split
        - Statistics: average earliest interaction gap per valid row
          (earliest gap = max real delta per row; real delta excludes -1 and max_span)
        """
        print("\n================ Final Verification ================")

        def verify_one(name: str, csv_path: str, npy_path: str):
            df = pd.read_csv(csv_path, usecols=[label_col])
            labels = df[label_col].to_numpy()

            arr = np.load(npy_path, allow_pickle=False)  # (M, N)
            if arr.shape[0] != labels.shape[0]:
                raise AssertionError(
                    f"[{name}] row mismatch: csv={labels.shape[0]} vs npy={arr.shape[0]}"
                )

            row_all_minus1 = np.all(arr == -1, axis=1)
            row_has_minus1 = np.any(arr == -1, axis=1)

            label0 = (labels == 0)
            label1 = ~label0

            # Rule checks
            bad0 = np.where(label0 & (~row_all_minus1))[0]
            bad1 = np.where(label1 & row_has_minus1)[0]
            if bad0.size > 0:
                print(f"[{name}] ERROR: label==0 but not all -1. examples={bad0[:10].tolist()}")
            if bad1.size > 0:
                print(f"[{name}] ERROR: label!=0 but contains -1. examples={bad1[:10].tolist()}")
            if bad0.size > 0 or bad1.size > 0:
                raise AssertionError(f"[{name}] label / -1 rule violated")

            # Valid rows (not all -1)
            valid_rows = int(np.sum(~row_all_minus1))
            label1_rows = int(np.sum(label1))

            # Real delta mask
            real_mask = (label1[:, None]) & (arr != -1) & (arr != max_span)

            real_cnt_per_row = np.sum(real_mask, axis=1)
            rows_with_history_mask = (label1 & (real_cnt_per_row > 0))
            rows_with_history = int(np.sum(rows_with_history_mask))

            # Oldest delta per row = max(real_deltas)
            neg_inf = np.float32(-np.inf) if arr.dtype == np.float32 else -np.inf
            arr_real_only = np.where(real_mask, arr, neg_inf)
            oldest_delta_per_row = np.max(arr_real_only, axis=1)

            if rows_with_history > 0:
                avg_oldest_delta = float(np.mean(oldest_delta_per_row[rows_with_history_mask]))
                max_oldest_delta = float(np.max(oldest_delta_per_row[rows_with_history_mask]))
            else:
                avg_oldest_delta = float("nan")
                max_oldest_delta = float("nan")

            # Newest delta per row = min(real_deltas)
            pos_inf = np.float32(np.inf) if arr.dtype == np.float32 else np.inf
            arr_real_only_min = np.where(real_mask, arr, pos_inf)
            newest_delta_per_row = np.min(arr_real_only_min, axis=1)
            if rows_with_history > 0:
                avg_newest_delta = float(np.mean(newest_delta_per_row[rows_with_history_mask]))
            else:
                avg_newest_delta = float("nan")

            print(f"\n[{name}]")
            print(f"  total rows                       : {arr.shape[0]}")
            print(f"  label!=0 rows                    : {label1_rows}")
            print(f"  valid rows (!all -1)             : {valid_rows}")
            print(f"  rows with >=1 real delta         : {rows_with_history}")
            print(f"  rows label!=0 but no history     : {label1_rows - rows_with_history}")
            print(f"  avg oldest delta (per-row max)   : {avg_oldest_delta}")
            print(f"  max oldest delta (across rows)   : {max_oldest_delta}")
            print(f"  avg newest delta (per-row min)   : {avg_newest_delta}")

            return {
                "valid_rows": valid_rows,
                "rows_with_history": rows_with_history,
                "avg_oldest_delta": avg_oldest_delta,
                "max_oldest_delta": max_oldest_delta,
                "avg_newest_delta": avg_newest_delta,
            }

        r_train = verify_one("train", train_csv, out_train_npy)
        r_val   = verify_one("val",   val_csv,   out_val_npy)
        r_test  = verify_one("test",  test_csv,  out_test_npy)

        print("\n[Summary: avg oldest delta (= earliest interaction gap)]")
        print(f"  train: rows_with_history={r_train['rows_with_history']}, avg_oldest_delta={r_train['avg_oldest_delta']}")
        print(f"  val  : rows_with_history={r_val['rows_with_history']},   avg_oldest_delta={r_val['avg_oldest_delta']}")
        print(f"  test : rows_with_history={r_test['rows_with_history']},  avg_oldest_delta={r_test['avg_oldest_delta']}")
        print("All final checks passed ✅")


    # Process in order: train -> val -> test (shared history)
    process_one(train_csv, out_train_npy)
    process_one(val_csv, out_val_npy)
    process_one(test_csv, out_test_npy)

    # Final verification
    final_verify()

    # Merge train, val, test into all
    train_data = np.load(out_train_npy, allow_pickle=True)
    val_data = np.load(out_val_npy, allow_pickle=True)
    test_data = np.load(out_test_npy, allow_pickle=True)
    all_data = np.concatenate([train_data, val_data, test_data], axis=0)
    np.save(out_all_npy, all_data)
    print(f"Merging completed! Saved to: {out_all_npy}")
    print(f"Total samples: {len(all_data)}")


if __name__ == "__main__":
    build_item_history_dt_npys(
        train_csv="../../processed_data/aminer/pp_aminer_half_year_train.csv",
        val_csv="../../processed_data/aminer/pp_aminer_half_year_val.csv",
        test_csv="../../processed_data/aminer/pp_aminer_half_year_test.csv",
        out_train_npy="../../processed_data/aminer/pp_aminer_half_year_train_dt_N200.npy",
        out_val_npy="../../processed_data/aminer/pp_aminer_half_year_val_dt_N200.npy",
        out_test_npy="../../processed_data/aminer/pp_aminer_half_year_test_dt_N200.npy",
        out_all_npy="../../processed_data/aminer/pp_aminer_half_year_all_dt_N200.npy",
        N=20,
        item_col="i",
        ts_col="ts",
        label_col="label",
        recent_first=True,
        out_dtype=np.float32,
        verify_show_k=10,
    )
