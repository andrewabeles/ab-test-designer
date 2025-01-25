# A/B Test Designer  
https://ab-test-designer.streamlit.app/

This is an app for estimating the sample size and amount of time needed to run an A/B test. It takes the following parameters as input: subjects per period, baseline, standard deviation (only for means), alpha, and power. It outputs the total sample size and number of periods required to detect with statistical significance a given delta between test and control.

## Subjects per Period
The number of subjects (users, devices, etc.) per week that will be included in the experiment.

## Metric Baseline
The baseline value of the variable being tested. The variable's proportion or average in the control group.

## Standard Deviation
The standard deviation of the baseline variable. Larger standard deviations require larger sample sizes to ensure the observed difference between groups is not due to the baseline variable's natural variance.

## Delta
The treatment's delta over the baseline. The absolute difference between the treatment and control.

## Alpha
Also known as the significance level, alpha is the experiment's Type I error (false positive) rate.

## Power
1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given delta at a given significance level.
