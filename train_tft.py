"""Train a TFT with disjoint time splits and persist its fitted preprocessing."""
import argparse
import copy
import os

from forecast_api.config import ENCODER_LENGTH, FORMAT_VERSION, MAX_HORIZON, QUANTILES, SEED, TFT_PATH, TFT_SPEC
from forecast_api.data import historical_data
from forecast_api.training import provenance, tft_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batches", type=int, default=100, help="Maximum batches per epoch; use 0 for all")
    parser.add_argument("--val-batches", type=int, default=30, help="Maximum validation batches; use 0 for all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or min(args.train_batches, args.val_batches) < 0:
        parser.error("Epochs, batch size, and threads must be positive; batch limits must be nonnegative.")

    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    class BestValidation(pl.Callback):
        def __init__(self):
            self.best = float("inf")
            self.state = None
            self.epoch = None

        def on_validation_end(self, trainer, model):
            value = trainer.callback_metrics.get("val_loss")
            if not trainer.sanity_checking and value is not None:
                print(f"Epoch {trainer.current_epoch + 1}: validation quantile loss={float(value):.4f}", flush=True)
            if not trainer.sanity_checking and value is not None and float(value) < self.best:
                self.best = float(value)
                self.epoch = trainer.current_epoch + 1
                self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(args.threads)
    history, digest = historical_data()
    training, validation, train_frame = tft_datasets(history)
    model = TemporalFusionTransformer.from_dataset(training, **TFT_SPEC, loss=QuantileLoss(quantiles=QUANTILES))
    best = BestValidation()
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.epochs,
        gradient_clip_val=0.1, limit_train_batches=args.train_batches or 1.0,
        limit_val_batches=args.val_batches or 1.0, deterministic=True,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        enable_checkpointing=False, callbacks=[best], num_sanity_val_steps=0,
    )
    trainer.fit(model,
                train_dataloaders=training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0),
                val_dataloaders=validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0))
    if best.state is None:
        raise RuntimeError("No finite validation loss was recorded; no artifact was replaced.")
    metadata = provenance("tft", history, digest)
    metadata.update(training_options=vars(args), best_epoch=best.epoch, validation_loss=best.best,
                    parameters=model.size(), quantile_rearrangement=True)
    bundle = {"format_version": FORMAT_VERSION, "metadata": metadata,
              "state_dict": best.state, "model_config": copy.deepcopy(TFT_SPEC),
              "quantiles": QUANTILES, "dataset_parameters": training.get_parameters(),
              "dataset_sample": train_frame.tail(ENCODER_LENGTH + MAX_HORIZON).copy()}
    TFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TFT_PATH.with_suffix(".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, TFT_PATH)
    print(f"Saved {TFT_PATH}; best epoch={best.epoch}, validation quantile loss={best.best:.4f}")


if __name__ == "__main__":
    main()
