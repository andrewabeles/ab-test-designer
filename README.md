# A/B Test Designer  
https://andrewabeles.shinyapps.io/power_calculator/

This is an app for estimating the sample size and amount of time needed to run an A/B test. It takes the following parameters as input: users per week, percent of traffic, baseline, standard deviation (only for means), alpha, and power. It outputs the total sample size and number of weeks required to detect with statistical significance a given lift over the baseline. If you find an experiment design that you like, you can give it a name and click 'Save' to record it in the History tab. You can then right click the history table to export it as a CSV.

## Users per Week
The number of users per week that will be included in the experiment.

## Percent of Traffic
The percent of traffic that will be included in the experiment. This is used to calculate the actual number of users per week that will be included.

## Baseline
The baseline value of the variable being tested. The variable's proportion or average in the control group.

## Standard Deviation
The standard deviation of the baseline variable. Larger standard deviations require larger sample sizes to ensure the observed difference between groups is not due to the baseline variable's natural variance.

## Lift
The treatment's lift over the baseline. The percentage difference between the treatment and control.

## Alpha
Also known as the significance level, alpha is the experiment's Type I error (false positive) rate.

## Power
1 - Beta, the Type II error or false negative rate. Power represents the probability the experiment will detect a given percent lift at a given significance level.
