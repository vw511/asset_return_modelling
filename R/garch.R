###### R functions to model asset returns with GARCH(1,1), comparing between 
###### truncated normal (TN) and mixture truncated normal (MTN) assumptions for log return

library(sn)
library(PerformanceAnalytics)
library(car)
library(tseries)
library(forecast)
library(quantmod)
library(rugarch)
library(FinTS)
library(zoo)
library(skewt)
require(skewt)
library(crch)
require(crch)
require(tidyverse)
library(fGarch)
library(truncnorm)


options(digits=4)

# ----------------- Predict return with GARCH(1,1) ----------------
garchfit = function(garch, Lag1){
  beta.val = rep(mean(Lag1), length(Lag1))
  for (i in 2:length(Lag1)){
    beta.val[i] = garch$par[1] + garch$par[2] * Lag1[i] + garch$par[3] * beta.val[i-1]
  }
  
  y.hat.g = garch$par[1] +
    garch$par[2]*Lag1 +
    garch$par[3]*beta.val
  return(y.hat.g)
}


# --------------- Loss functions-----------------
# Truncated normal loss function
GARCHfit_tn = function(params,x,y){
  omega = params[1]
  alphaL1 = params[2]
  betaL1 = params[3]
  n = length(y)
  
  sigma2t = rep(mean(x),n)
  for (i in 2:n){
    sigma2t[i] = omega + alphaL1 * x[i] + betaL1 * sigma2t[i-1]}
  
  sigmat = sqrt(sigma2t)
  log.likelihood = -sum(log(dtruncnorm(y,lb,ub,0,sigmat) ))
  return(log.likelihood)
}

# Mixture truncated normal loss function
GARCHfit_mtn = function(params,x,y){
  omega = params[1]
  alphaL1 = params[2]
  betaL1 = params[3]
  n = length(y)
  
  sigma2t = rep(mean(x),n)
  for (i in 2:n){
    sigma2t[i] = omega + alphaL1 * x[i] + betaL1 * sigma2t[i-1]
  }
  sigmat = sqrt(sigma2t)
  sigmatt_1 = sd[1]/(sd[1]+sd[2]+sd[3])*sigmat
  sigmatt_2 = sd[2]/(sd[1]+sd[2]+sd[3])*sigmat
  sigmatt_3 = sd[1]/(sd[1]+sd[2]+sd[3])*sigmat
  
  # log.likelihood = -sum(log(dnorm(y, 0, sigmat)))
  log.likelihood = -sum(log(pie[1]*dtruncnorm(y,lb,ub,muu[1],sigmatt_1)
                            +pie[2]*dtruncnorm(y,lb,ub,muu[2],sigmatt_2)
                            +pie[3]*dtruncnorm(y,lb,ub,muu[3],sigmatt_3)))
  return(log.likelihood)
}

