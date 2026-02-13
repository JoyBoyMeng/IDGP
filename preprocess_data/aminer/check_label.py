import pandas as pd
import numpy as np

CSV_TRAIN = "../../processed_data/aminer/pp_aminer_half_year_train.csv"
CSV_VAL   = "../../processed_data/aminer/pp_aminer_half_year_val.csv"
CSV_TEST  = "../../processed_data/aminer/pp_aminer_half_year_test.csv"

LABEL_COL = "label"


def compute_avg_label(csv_path: str, split_name: str):
    df = pd.read_csv(csv_path, usecols=[LABEL_COL])
    labels = df[LABEL_COL].to_numpy()

    valid_mask = (labels != 0)
    valid_cnt = int(valid_mask.sum())

    if valid_cnt > 0:
        avg_label = float(np.mean(labels[valid_mask]))
    else:
        avg_label = float("nan")

    print(f"[{split_name}]")
    print(f"  total rows        : {len(labels)}")
    print(f"  label!=0 rows     : {valid_cnt}")
    print(f"  avg(label | label!=0) : {avg_label}\n")

    return avg_label, valid_cnt


def main():
    compute_avg_label(CSV_TRAIN, "train")
    compute_avg_label(CSV_VAL,   "val")
    compute_avg_label(CSV_TEST,  "test")


if __name__ == "__main__":
    main()
