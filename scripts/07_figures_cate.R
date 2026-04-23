#!/usr/bin/env Rscript
# 07_figures_cate.R
# -----------------
# Generates supplementary ATE/CATE comparison plots (Web Appendix).
#
# Inputs (from output/tables/simulate_base/):
#   ate_sample500_tau1_0_0.1_tau2_0_sim100.csv
#
# Usage:
#   Rscript scripts/07_figures_cate.R

library(ggplot2)
library(data.table)
setwd("../output/tables")  # adjust if needed

ate_data_s_sim=read.csv("sim_dense_ate_sample500_tau_small.csv")
ate_data_s=read.csv("sim_dense_ate_sample500_tau_small_sim500.csv")
ate_data_s=rbind(ate_data_s[ate_data_s$estimator != "representative",], ate_data_s_sim[ate_data_s_sim$estimator=="representative",])
# ate_data_s=read.csv("sim_dense_ate_sample1000_tau_small.csv")
ate_data_s$estimand <- factor(ate_data_s$estimand, levels = c('d10', 'd01'))
ate_data_s$abs_bias=abs(ate_data_s$bias)
ate_data_s=data.table(ate_data_s)
ate_data_s_summary=ate_data_s[estimand=="d10", lapply(.SD, mean) ,estimator, .SDcols = c("true_value","avg_est", "avg_se", "sd_est", "mse", "power", "coverage", "abs_bias")]

ate_data_l=read.csv("sim_dense_ate_sample500_tau_large.csv")
# ate_data_l=read.csv("sim_dense_ate_sample1000_tau_large.csv")
# ate_data_l=ate_data_l[ate_data_l$tau1>=0.2,]
ate_data_l$abs_bias=abs(ate_data_l$bias)
ate_data_l=data.table(ate_data_l)
ate_data_l_summary=ate_data_l[estimand=="d10", lapply(.SD, mean) ,estimator, .SDcols = c("true_value","avg_est", "avg_se", "sd_est", "mse", "power", "coverage", "abs_bias")]


ate_data=rbind(ate_data_s[ate_data_s$tau1 %in% c(0, 0.1),], ate_data_l)
ate_data$estimand <- factor(ate_data$estimand, levels = c('d10', 'd01'))
table(ate_data$tau1)

#### mse and bias ratio show the efficiency of the estimators
ggplot(ate_data_s, aes(x=tau1, y=mse,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand, scales = "free_y")
ggplot(ate_data_s[ate_data_s$estimator %in% c("diff-in-mean",'representative','Hajek'),], aes(x=tau1, y=mse,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand, scales = "free_y")
### abs(bias)
ggplot(ate_data_s, aes(x=tau1, y=abs(bias),  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand, scales = "free_y")
ggplot(ate_data_s, aes(x=tau1, y=abs(bias/true_value),  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand, scales = "free_y")

#### power and coverage are more meaningful when tau1 is relatively small
ggplot(ate_data_s, aes(x=tau1, y=power,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand)

ggplot(ate_data_s, aes(x=tau1, y=coverage,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 0.1, by = 0.01), labels = seq(0, 0.1, by = 0.01))+
  facet_wrap(~estimand)

#### Large tau: mse and bias ratio show the efficiency of the estimators
ggplot(ate_data, aes(x=tau1, y=mse,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")
ggplot(ate_data[ate_data$estimator %in% c("diff-in-mean",'representative'),], aes(x=tau1, y=mse,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

ggplot(ate_data, aes(x=tau1, y=bias,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

# summary(lm(abs(bias)~tau1+tau1*estimator, data=ate_data_s[estimand=="d10"]))
ggplot(ate_data[tau1>0], aes(x=tau1, y=abs(bias),  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0.1, 1.1, by = 0.1), labels = seq(0.1, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

ggplot(ate_data[tau1>0], aes(x=tau1, y=abs(bias/true_value),  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0.1, 1.1, by = 0.1), labels = seq(0.1, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

ggplot(ate_data[!ate_data$estimator=='Horvitz-Thompson',], aes(x=tau1, y=abs(bias/true_value),  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

ggplot(ate_data, aes(x=tau1, y=power,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")

ggplot(ate_data, aes(x=tau1, y=coverage,  color=estimator))+
  geom_line()+
  scale_x_continuous(breaks = seq(0, 1.1, by = 0.1), labels = seq(0, 1.1, by = 0.1))+
  facet_wrap(~estimand, scales = "free_y")


### show tables
View(ate_data[tau1==0.1,])
View(ate_data[round(tau1,1)==0.5,])



#### CATE

