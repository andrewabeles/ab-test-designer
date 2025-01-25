import streamlit as st 
import pandas as pd 
import numpy as np
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power
import plotly.express as px

def estimate_sample_sizes(control_mean, control_std=None, deltas=None, metric_type='proportion', alpha=0.05, power=0.8, alternative='two-sided'):
    results = {
        'test_mean': [],
        'delta': [],
        'effect_size': [],
        'total_sample_size': []
    }

    for d in deltas:
        test_mean = control_mean + d 
        if metric_type == 'proportion':
            effect_size = proportion_effectsize(test_mean, control_mean)
            sample_size_per_group = zt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative=alternative
            )
        elif metric_type == 'mean':
            effect_size = d / control_std
            sample_size_per_group = tt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative=alternative
            )
        
        # handle edge-case where statsmodels returns [10.0] instead of very small sample sizes
        if type(sample_size_per_group) != float: 
            sample_size_per_group = 1
        
        total_sample_size = np.ceil(sample_size_per_group) * 2
        
        results['test_mean'].append(test_mean)
        results['delta'].append(d)
        results['effect_size'].append(effect_size)
        results['total_sample_size'].append(total_sample_size)

    results['metric_type'] = metric_type
    results['alternative_hypothesis'] = alternative
    results['alpha'] = alpha
    results['power'] = power 
    results['control_mean'] = control_mean
    results['control_std'] = control_std

    return pd.DataFrame(results)

st.title("A/B Test Designer")

with st.expander("About"):
    st.write("""
        Use this app to help design and plan A/B tests. Start by entering the experiment's details in the sidebar. 
        Then use the chart to see the minimum delta between the test and control groups the experiment would be
        able to detect with statistical significance after a given number of periods. Use this information to 
        understand how long the A/B test would have to run to provide the insights needed to make a decision. 
        Download the raw data for reference and sharing.   
    """)

with st.sidebar:
    subjects_per_period = st.number_input(
        "Subjects per Period", 
        min_value=1, 
        value=1000,
        help="""Number of unique subjects (users, devices, etc.) you expect to enter the test each period (day, week, etc.)."""
    )

    metric_type = st.radio(
        "Metric Type", 
        ["proportion", "mean"],
        help="""Select 'proportion' if the test's success metric is a conversion rate, and 'mean' if it's an average (e.g. revenue per user)."""
    )

    control_mean = st.number_input(
        "Metric Baseline", 
        min_value=0.0 if metric_type == 'proportion' else 0, 
        max_value=1.0 if metric_type == 'proportion' else None,
        value=0.1 if metric_type == 'proportion' else 10,
        help="""Expected value of the success metric for the control group."""
    )

    if metric_type == 'proportion':
        control_std = np.sqrt(control_mean * (1-control_mean))
    else:
        control_std = st.number_input(
            "Metric Standard Deviation", 
            min_value=0, 
            value=3,
            help="""Expected standard deviation of the success metric for the control group."""
        )

    alternative = st.selectbox(
        "Alternative Hypothesis", 
        ["two-sided", "larger", "smaller"],
        help="""Select 'two-sided' if testing for an impact on the success metric in either direction.
                Select 'larger' if only testing for an increase.
                Select 'smaller' if only testing for a decrease."""
    )

    alpha = st.selectbox(
        "Alpha", 
        [0.01, 0.05, 0.1, 0.2], 
        index=1,
        help="""False Positive Rate. The probability the test will be statistically significant merely by chance."""
    )

    power = st.selectbox(
        "Power", 
        [0.8, 0.85, 0.9, 0.95],
        help="""1 - False Negative Rate. The probability the test will detect a minimum effect size with statistical significance if it truly exists."""
    )

min_delta = 0
max_delta = 2 * control_mean 
if metric_type == 'proportion':
    max_delta = 1 - control_mean
if alternative == 'smaller':
    max_delta = -1 * max_delta
deltas = np.linspace(min_delta, max_delta, num=100)[1:]

results = estimate_sample_sizes(
    control_mean,
    control_std=control_std,
    deltas=deltas,
    metric_type=metric_type,
    alpha=alpha,
    power=power,
    alternative=alternative
)

results['subjects_per_period'] = subjects_per_period 
results['periods'] = results['total_sample_size'] / results['subjects_per_period']

fig = px.line(
    results,
    y='delta',
    x='periods',
    markers=True,
    title='Minimum Detectable Delta by Test Duration'
)

st.plotly_chart(fig)

results[[
    'metric_type',
    'alternative_hypothesis',
    'alpha',
    'power',
    'control_mean',
    'control_std',
    'test_mean',
    'delta',
    'effect_size',
    'total_sample_size',
    'subjects_per_period',
    'periods'
]]
