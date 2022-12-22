###### R functions to model asset returns with LSTM, comparing between 
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

# ---------Activation function and forward propagation-----------
actfun = function(x){x}

sigmoid = function(x){exp(x)/(1+exp(x))}

lstm.fwdprop = function (x,w){
  omega = w[1]; alpha = w[2]; beta = w[3]
  wf = w[4]; af = w[5]; bf = w[6]
  wi = w[7]; ai = w[8]; bi = w[9]
  wo = w[10]; ao = w[11]; bo = w[12]
  
  sigma2t = rep(mean(x),nrow(x))
  ot = rep(0, nrow(x))
  ft = rep(0, nrow(x))
  it = rep(0, nrow(x))
  Ct = rep(0, nrow(x))
  
  for (k in 2:nrow(x)){
    ot[k] = sigmoid(wo + ao * x[k] + bo * sigma2t[k-1])
    ft[k] = sigmoid(wf + af * x[k] + bf * sigma2t[k-1])
    it[k] = sigmoid(wi + ai * x[k] + bi * sigma2t[k-1])
    Ct[k] = ft[k]*Ct[k-1] + it[k]*( omega + alpha*x[k] + beta*sigma2t[k-1])
    sigma2t[k] = ot[k] * Ct[k]
  }
  
  y = sigma2t # identity function output
  list (output = y)
}
  

# --------------- Loss functions-----------------

# Truncated normal loss function
loss.fun_tn = function (init.w,x,y,l,u){
  y.hat = lstm.fwdprop(x,init.w)$output
  n = length(y)
  return(-sum(log(truncnorm::dtruncnorm(y, l, u, 0, sqrt(y.hat)))))
}

pie = c(0.96649, 0.01786, 0.01565)
muu = c(-0.03883, -11.0972, 10.49614 )
sd = c(4.5141, 1.9978, 0.3079) # it is variance
vec = c(pie,muu,sd)
  
# Mixture truncated normal loss function
loss.fun_mtn = function (init.w,x,y,l,u,pie,mu,sd){
  y.hat = lstm.fwdprop(x,init.w)$output
  n = length(y)
  sum_sd = sum(sd)
    
  prob_sum = 0
  for (i in 1:length(pie)) {
    prob_sum = prob_sum + pie[i]*truncnorm::dtruncnorm(y,l,u,mu[i],sd[i]/sum_sd*sqrt(y.hat))
  }
  
  return(-sum(log(prob_sum)))
}

