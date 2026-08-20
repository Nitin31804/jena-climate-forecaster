# train_tft.py
"""
Train a Temporal Fusion Transformer (TFT) on the hourly Jena Climate data.
The script:
1. Loads df_hourly.csv with correct 'Date Time' column.
2. Performs a simple train/validation split (last 720 rows for validation).
3. Defines a TimeSeriesDataSet for PyTorch Forecasting.
4. Trains the TFT for 2 epochs on CPU.
5. Saves the trained model state dict to tft_checkpoint.pth.
"""

import sys

try:
    import pandas as pd
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    DATA_PATH = r"C:\Users\Admin\OneDrive\Documents\df_hourly.csv"
    CHECKPOINT_PATH = r"C:\Users\Admin\OneDrive\Documents\tft_checkpoint.pth"
    TARGET = "T (degC)"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data from: " + DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date Time"])
    print("Loaded dataframe with shape: " + str(df.shape))

    # ------------------------------------------------------------------
    # 2. Create integer time_idx column (row number)
    # ------------------------------------------------------------------
    df = df.reset_index(drop=True)
    df["time_idx"] = df.index

    # ------------------------------------------------------------------
    # 3. Add constant group_id column (string)
    # ------------------------------------------------------------------
    df["series_id"] = "0"

    # Keep only needed columns
    df = df[["time_idx", "series_id", TARGET]].copy()
    print("Columns kept: " + str(list(df.columns)))
    print("First 3 rows:")
    print(df.head(3).to_string())

    # ------------------------------------------------------------------
    # 4. Train/validation split: last 720 rows for validation
    # ------------------------------------------------------------------
    max_idx = df["time_idx"].max()
    val_cutoff = max_idx - 720
    train_df = df[df["time_idx"] <= val_cutoff].copy()
    val_df = df[df["time_idx"] > val_cutoff].copy()
    print("Train size: " + str(len(train_df)) + " | Val size: " + str(len(val_df)))

    # ------------------------------------------------------------------
    # 5. Define TimeSeriesDataSet
    # ------------------------------------------------------------------
    max_encoder_length = 168
    max_prediction_length = 168

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[TARGET],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df)

    # Dataloaders
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    print("Dataloaders created successfully.")

    # ------------------------------------------------------------------
    # 6. Model definition (small model for quick demo)
    # ------------------------------------------------------------------
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=2,
    )
    print("TFT model created. Parameters: " + str(tft.size()))

    # ------------------------------------------------------------------
    # 7. Trainer from pytorch_lightning (CPU only)
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        gradient_clip_val=0.1,
        limit_train_batches=20,
        limit_val_batches=5,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    print("Starting training...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")

    # ------------------------------------------------------------------
    # 8. Save model state dict
    # ------------------------------------------------------------------
    torch.save(tft.state_dict(), CHECKPOINT_PATH)
    print("Model state dict saved to: " + CHECKPOINT_PATH)
    print("TFT training completed successfully.")

except Exception as e:
    print("ERROR during TFT training: " + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
