###### R functions to calculate log returns and realized volatility from raw data

library(readr)
library(PerformanceAnalytics)
library(zoo)

# Helper: Calculate realized volatility of a single row of 15-min data
calc_realized_vol <- function(row) {
  len = length(row) - 1
  prices = row[1:len+1]
  log_rtns_sqr = log(prices[-1]/prices[-len], base=exp(1))^2
  rz = sqrt(sum(log_rtns_sqr) / length(log_rtns_sqr))
  return(rz)
}

# Construct daily realized volatility for all given days of a stock
get_realized_vols <- function(data) {
  n = nrow(data)
  vols = rep(0, n)
  for (i in 1:n){
    vols[i] = calc_realized_vol(data[i,])
  }
  return(vols)
}

# Create and save calculated data to new csv in the 'data/' folder. 
# - raw_csv: the input csv
# - file_name: the output csv
# Columns in the output file include:
# - date
# - close
# - constructed datily realized volatility
# - log return
create_csv <- function(raw_csv, file_name) {

  data = read_csv(raw_csv)
  
  data_zoo = read.csv.zoo(raw_csv, header=TRUE, na.strings="null")
  close = data_zoo[,"close", drop=F]
  close = na.omit(close)
  rtn = as.xts(CalculateReturns(close, method="log"))
  rtn = rtn[-1]
  
  df <- data.frame (date = data$date[-1], close = data$close[-1], 
                    realized_vol = get_realized_vols(data)[-1], 
                    log_rtn = as.vector(coredata(rtn)))
  
  # write.csv(df, paste('data/', file_name, sep=""))
  write.csv(df, file_name)
}

create_csv('000923_raw.csv', 'SZ000923.csv')
create_csv('600158_raw.csv', 'SH600158.csv')
create_csv('600679_raw.csv', 'SH600679.csv')







