import numpy as np
import pytest

from forecast_api.config import ENCODER_LENGTH, FORMAT_VERSION, MAX_HORIZON, QUANTILES, TFT_SPEC
from forecast_api.data import prediction_frame, split_boundaries
from forecast_api.models import ModelUnavailable, load_tft, predict_tft, validate_prediction
from forecast_api.training import provenance, tft_datasets


@pytest.mark.parametrize("values", [([1], [0], [2, 3]), ([np.nan], [0], [2]), ([1], [2], [3])])
def test_invalid_model_outputs_rejected(values):
    with pytest.raises(ModelUnavailable):
        validate_prediction(values, 1)


def test_sarima_parameter_roundtrip_and_malformed_artifact(tmp_path, history):
    from forecast_api.config import SARIMA_SPEC
    from forecast_api.models import atomic_json, load_sarima, predict_sarima

    path = tmp_path / "sarima.json"
    meta = provenance("sarima", history, "test")
    meta["converged"] = True
    payload = {"format_version": FORMAT_VERSION, "metadata": meta, "spec": SARIMA_SPEC,
               "parameters": [0.1, 0.2, 0.5, 0.1, 1.0]}
    atomic_json(path, payload)
    loaded = load_sarima(path)
    assert load_sarima(path) is loaded
    for horizon in [1, 24, 168]:
        validate_prediction(predict_sarima(loaded, history, horizon), horizon)
    payload["parameters"] = [1.0]
    atomic_json(path, payload)
    with pytest.raises(ModelUnavailable):
        load_sarima(path)


@pytest.mark.slow
def test_tft_roundtrip_and_decoder_boundaries(tmp_path, history):
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    torch.set_num_threads(2)
    train_end, _ = split_boundaries(len(history))
    history.loc[train_end + 50, "temperature_observed"] = False
    training, validation, sample = tft_datasets(history)
    assert training.decoded_index.time_idx_last.max() < train_end
    assert validation.decoded_index.time_idx_first_prediction.min() >= train_end
    assert not ((validation.decoded_index.time_idx_first_prediction <= train_end + 50) &
                (validation.decoded_index.time_idx_last >= train_end + 50)).any()
    model = TemporalFusionTransformer.from_dataset(training, **TFT_SPEC, loss=QuantileLoss(quantiles=QUANTILES))
    model.eval()
    path = tmp_path / "tft.pt"
    torch.save({"format_version": FORMAT_VERSION, "metadata": provenance("tft", history, "test"),
                "state_dict": model.state_dict(), "model_config": TFT_SPEC, "quantiles": QUANTILES,
                "dataset_parameters": training.get_parameters(),
                "dataset_sample": sample.tail(ENCODER_LENGTH + MAX_HORIZON)}, path)
    loaded = load_tft(path)
    assert load_tft(path) is loaded
    frame = prediction_frame(history)
    dataset = TimeSeriesDataSet.from_dataset(loaded[1], frame, predict=True, stop_randomization=True)
    assert dataset.decoded_index.time_idx_first_prediction.iloc[0] == len(history)
    assert not set(loaded[0].hparams.time_varying_reals_decoder) & {"T (degC)", "temperature_mean_6h", "temperature_mean_24h"}
    before = predict_tft((model, training, {}, QUANTILES), history, 24)
    after = predict_tft(loaded, history, 24)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=1e-6)
    for horizon in [1, 168]:
        validate_prediction(predict_tft(loaded, history, horizon), horizon)
