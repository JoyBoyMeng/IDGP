# IDGP: Influence-Aware Dynamic Graph Learning for Popularity Prediction

This repository contains the official implementation of **IDGP**, an
influence-aware dynamic graph framework for cascade popularity
prediction.

IDGP models long-range temporal influence by dynamically routing node
representations to evolving latent influence carriers (*fluxons*),
enabling global trend aggregation beyond local neighborhood propagation.

------------------------------------------------------------------------

## Dataset

The dataset used in this project can be found in:

> *Continuous-Time Graph Learning for Cascade Popularity Prediction*

Please use the tools provided in the `preprocess_data` directory to
preprocess the dataset step by step before training.

------------------------------------------------------------------------

## Training

To train the IDGP model:

```bash
python train_popularity_prediction.py \
    --dataset_name aminer \
    --model_name Fluxion \
    --num_runs 3 \
    --patience 20 \
    --batch_size 200 \
    --learning_rate 0.0001 \
    --gpu 0 \
    --fluxion_member_num 64 \
    --ema 0.5 \
    --num_epochs 30 \
    --exp 001
```

------------------------------------------------------------------------

## Evaluation

To evaluate the trained model:

```bash
python evaluate_popularity_prediction.py \
    --dataset_name aminer \
    --model_name Fluxion \
    --num_runs 1 \
    --batch_size 200 \
    --gpu 0 \
    --fluxion_member_num 64 \
    --ema 0.5 \
    --exp 001
```

------------------------------------------------------------------------

## Metrics

The framework reports the following metrics:

-   **MSLE**
-   **MALE**
-   **MAPE**
-   **PCC**

Performance is averaged across multiple runs.

