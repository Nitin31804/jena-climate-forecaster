# Held-out temperature evaluation

Generated 2026-09-16T16:46:15.165867+00:00. 12 disjoint windows, 24 hours each.

Models use frozen parameters; all targets follow training and validation. SARIMA conditions on up to 720 past hours; TFT uses 168 past hours. Only calendar covariates are supplied for future hours.

| Model                |   MAE (°C) |   RMSE (°C) |
|:---------------------|-----------:|------------:|
| Persistence          |      2.354 |       3.133 |
| Seasonal naive (24h) |      2.879 |       3.710 |
| SARIMA               |      2.228 |       2.863 |
| TFT                  |      2.469 |       3.257 |

Lower is better. These sampled windows do not establish general superiority. TFT training settings, model versions, per-window errors, and interval coverage are recorded in backtest_results.json. Celsius percentage errors are intentionally omitted.
