###### Quick demo: 
# - run em to get the paramters for mixture truncated normal (MTN)
# - feed the trained parameters from em for the loss function in LSTM and GARCH
# - compare LSTM and GARCH results, and compare betweem the MTN and TN assumptions

source('R/em.R')
source('R/lstm.R')
source('R/garch.R')

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
library(Rsolnp)

options(digits=4)

#============== EM ================#

# Use SZ000006's log return as data
dat = read_csv('data/SZ000006.csv')
dat.ret = dat$log_rtn
Y = 100 * dat.ret

mu0 <- c(0, -10, 10)
Sigma0 <- c(1, 1, 1)
pi0 <- c(0.8, 0.1, 0.1)
EM_estimate <- EM_mix_trunc_normal(Y, K=3, pi0, mu0, Sigma0, 11, 0.0001)

print("------------- EM results: ----------------")
print("Final Results for SZ000006's Log Return (3 components):")
print(c("Bounds of current data:", min(Y), max(Y)))
print(c('Number of data points:', length(Y)))
print(c('initial guess for the means:', mu0))
print(c('initial guess for the variances:', Sigma0))
print(c('initial guess for the latent variables:', pi0))
print(c('Output means:', EM_estimate$mu))
print(c('Output variances:', EM_estimate$Sigma))
print(c('Output weights:', EM_estimate$pi))
print(c('Iterations:', EM_estimate$Iter))
print(c('Log Likelihood:', EM_estimate$Q))


#============== LSTM ================#

dat = read.csv.zoo("data/SZ000006.csv", header=TRUE, na.strings="null")
dat = na.omit(dat)
dat = dat[,"close", drop=F]
dat.ret = as.xts(CalculateReturns(dat, method="log"))
dat.ret = dat.ret[-1]
colnames(dat.ret) = "return"

# Chinese stock boundaries (simplified approach)
ub = max(dat.ret)
lb = min(dat.ret)

# Split data
# nsplit.dat = which(dat$date == "2019-03-29")
nsplit.dat = which(time(dat.ret[,1]) == "2019-03-29")
dat.train = dat.ret[1:nsplit.dat]
x = (dat.train^2)[-length(dat.train)]
y = dat.train[-1]
init.k = rep(1/length(y),length(y))

# Use em results to define loss function
pie = EM_estimate$pi
muu = EM_estimate$mu
sd = EM_estimate$Sigma # this is variance
vec = c(pie,muu,sd)
# pie = c(0.96649, 0.01786, 0.01565)
# muu = c(-0.03883, -11.0972, 10.49614 )
# sd = c(4.5141, 1.9978, 0.3079) # it is variance
# vec = c(pie,muu,sd)

# TN LSTM engine
cl = makePSOCKcluster(2)
clusterExport(cl, c("sigmoid","loss.fun_tn","lstm.fwdprop"))
lstm_tn = gosolnp(fun = loss.fun_tn,  x = x, y = y, u=ub, l=lb,
                  LB = c(0,0,0,rep(c(-5,-1e3,-1e3),3)),
                  UB = c(0.05,1,1,rep(c(5,1e3,1e3),3)),
                  cluster = cl,
                  rseed = 20811644) 
stopCluster(cl)

# MTN LSTM engine
cl = makePSOCKcluster(2)
clusterExport(cl, c("sigmoid","loss.fun_mtn","lstm.fwdprop"))
lstm_mtn = gosolnp(fun = loss.fun_mtn,  x = x, y = y, u=ub, l=lb,
                   pie = pie, mu = muu, sd = sd,
                   LB = c(0,0,0,rep(c(-5,-1e3,-1e3),3)),
                   UB = c(0.05,1,1,rep(c(5,1e3,1e3),3)),
                   cluster = cl,
                   rseed = 20811644)
stopCluster(cl)

# Fitted LSTM with MTN and TN  
y.lstm_tn = lstm.fwdprop(x = dat.ret[-length(dat.ret)]^2, w = lstm_tn$pars)
y.lstm_mtn = lstm.fwdprop(x = dat.ret[-length(dat.ret)]^2, w = lstm_mtn$pars)



#============== GARCH(1,1) ================#
# Optimize (user-defined) losses
init.guess=c(0,rep(0.5,2))
garch_tn = optim(par = init.guess, fn=GARCHfit_tn,
                 x = (dat.train^2)[-length(dat.train)],
                 y = dat.train[-1],)

garch_mtn = optim(par = init.guess, fn=GARCHfit_mtn,
                  x = (dat.train^2)[-length(dat.train)],
                  y = dat.train[-1],)

# Fit GARCH(1,1) with MTN and TN
y.garch_tn = garchfit(garch = garch_tn, Lag1 = dat.ret[-length(dat.ret)]^2 )
y.garch_mtn = garchfit(garch = garch_mtn, Lag1 = dat.ret[-length(dat.ret)]^2 )



#============== Plotting comparisons ================#
print('Finished fitting LSTM and GARCH. Showing plots:')

###### Plot return volatility forecasts by LSTM under MTN and TN
png("graphs/lstm(mtn_vs_tn).png")
par(mfrow=c(1,1))
plot(time(dat.ret[-1]),sqrt(as.numeric(unlist(y.lstm_tn)))*100,col="red",lwd=3,type="l",
     ylim=c(0,30),ylab="Return volatility forecasts",
     xlab="Time", main="Volatility of LSTM under MTN and TN (000006.SZ)")
par(new=TRUE)
plot(time(dat.ret[-1]),sqrt(as.numeric(unlist(y.lstm_mtn)))*100,col="black",lwd=1,type="l",
     ylim=c(0,30),ylab="Return volatility forecasts",
     xlab="Time")
legend("topleft",c("LSTM_MTN","LSTM_TN"),col=c("black","red"),lty=c(2,1))
abline(v=as.numeric(nsplit.dat),lwd=2)
dev.off()
browseURL("graphs/lstm(mtn_vs_tn).png")


###### Plot return volatility forecasts by GARCH under MTN and TN
png("graphs/garch(mtn_vs_tn).png")
par(mfrow=c(1,1))
plot(sqrt(as.numeric(unlist(y.garch_tn)))*100,col="red",lwd=3,type="l",
     ylim=c(0,15),ylab="Return volatility forecasts",
     xlab="Time", main= "Volatility of GARCH(1,1) under MTN and TN (000006.SZ)",)
points(sqrt(as.numeric(y.garch_mtn))*100,type="l",lty=2)
legend(x=0,y=15,c("Garch(1,1)","LSTM"),col=c("black","red"), lty=c(2,1))
dev.off()
browseURL("graphs/garch(mtn_vs_tn).png")


##### Compare GARCH and LSTM under TN
png("graphs/lstm_vs_garch(tn).png")
par(mfrow=c(1,1))
plot(time(dat.ret[-1]),sqrt(as.numeric(unlist(y.lstm_tn)))*100,col="red",lwd=3,type="l",
     ylim=c(0,20),ylab="Return volatility forecasts", main="Volatility of LSTM vs GARCH(1,1) under TN", 
     xlab="Time")
par(new=TRUE)
plot(time(dat.ret[-1]),sqrt(as.numeric(y.garch_tn))*100,col="black",lwd=1,type="l",
     ylim=c(0,20),ylab="Return volatility forecasts",
     xlab="Time", main="Volatility of LSTM vs GARCH(1,1) under TN")
legend("topleft",c("GARCH(1,1) TN","LSTM TN"),col=c("black","red"),lty=c(2,1))
abline(v=as.numeric(nsplit.dat),lwd=2)
dev.off()
browseURL("graphs/lstm_vs_garch(tn).png")


##### Compare  GARCH and LSTM under MTN
png("graphs/lstm_vs_garch(mtn).png")
par(mfrow=c(1,1))
plot(time(dat.ret[-1]),sqrt(as.numeric(unlist(y.lstm_mtn)))*100,col="red",lwd=3,type="l",
     ylim=c(0,30),ylab="Return volatility forecasts",
     xlab="Time", main="Volatility of LSTM vs GARCH(1,1) under MTN")
par(new=TRUE)
plot(time(dat.ret[-1]),sqrt(as.numeric(y.garch_mtn))*100,col="black",lwd=1,type="l",
     ylim=c(0,30),ylab="Return volatility forecasts",
     xlab="Time", main="Volatility of LSTM vs GARCH(1,1) under MTN")
legend("topleft",c("GARCH(1,1) MTN","LSTM MTN"),col=c("black","red"),lty=c(2,1))
abline(v=as.numeric(nsplit.dat),lwd=2)
dev.off()
browseURL("graphs/lstm_vs_garch(mtn).png")










