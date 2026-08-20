# Backtest Results -- SARIMA(1,0,1) Walk-Forward

- **Model**: ARMA(1,0,1) -- no differencing, no seasonal component
- **Training window**: 500 hours
- **Forecast horizon**: 24 hours
- **Number of windows**: 5

## Per-Window Results

|   Window | Train Start         | Forecast Start      |   MAE (degC) |   RMSE (degC) |   MAPE (%) |
|---------:|:--------------------|:--------------------|-------------:|--------------:|-----------:|
|        1 | 2009-01-01 23:00:00 | 2009-01-22 19:00:00 |       1.4546 |        2.0157 |    84.0399 |
|        2 | 2010-12-27 18:00:00 | 2011-01-17 14:00:00 |       1.7639 |        2.1327 |    48.6422 |
|        3 | 2012-12-21 14:00:00 | 2013-01-11 10:00:00 |       0.6778 |        0.8261 |    79.3065 |
|        4 | 2014-12-16 09:00:00 | 2015-01-06 05:00:00 |       1.4869 |        1.7454 |   134.871  |
|        5 | 2016-12-10 05:00:00 | 2016-12-31 01:00:00 |       2.4172 |        3.3386 |   165.982  |

## Average Metrics

| Metric | Value |
|--------|-------|
| MAE (degC) | 1.5601 |
| RMSE (degC) | 2.0117 |
| MAPE (%) | 102.5684 |
