# asset_return_modelling
Modelling Asset Returns under Price Limits with Mixture of Truncated Normal GARCH and Deep Learning Architectures

## Quick Start

```
git clone https://github.com/vw511/asset_return_modelling.git
cd asset_return_modelling
Rscript 'demo.R'
```

Alternatively, download the `.RMD` files and run each chunk locally.

## data

This directory includes the following:

- `data_8raw.h5` contains information of eight Chinese stocks
- `XXXXXX_raw.csv` contains 15-minute prices from 9:00 (open), 9:15, ..., 11:30, 13:15, ..., 14:45, 15:00 (close). 
  - Prices in these files are extracted from `data_8raw.h5` using code in `extract_raw_prices.ipynb`
- `SXXXXXXX.csv` contains the trading dates, close prices, calculated daily log returns and constructed daily volatility of a particular stock.
  - Information in these files are extracted and calculated using code in `calculate_data_from_raw_prices.R`

## R

This directory includes user defined R functions that feature this project.

(For the final paper related to this project, please contact the owner of this GitHub repository)
