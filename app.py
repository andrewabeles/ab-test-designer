import streamlit as st 
import pandas as pd 
import numpy as np
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power
import plotly.express as px

def get_min_detectable_difs(control_mean, control_std=None, subjects_per_period=1000, max_periods=10, metric_type='proportion', alpha=0.05, power=0.8, alternative='two-sided'):
    results = {
        'test_mean': [],
        'means_dif': [],
        'effect_size': [],
        'total_sample_size': [],
        'periods': []
    }

    for p in range(1, max_periods+1):
        total_sample_size = p * subjects_per_period 
        sample_size_per_group = total_sample_size / 2
        if metric_type == 'proportion':
            effect_size = zt_ind_solve_power(
                nobs1=sample_size_per_group,
                alpha=alpha,
                power=power,
                alternative='larger' if alternative == 'smaller' else alternative,
                effect_size=None
            )
        elif metric_type == 'mean':
            effect_size = tt_ind_solve_power(
                nobs1=sample_size_per_group,
                alpha=alpha,
                power=power,
                alternative='larger' if alternative == 'smaller' else alternative,
                effect_size=None
            )
        
        # handle edge case where statsmodels returns iterable
        try:
            effect_size = effect_size[0]
        except:
            pass

        if alternative == 'smaller':
            effect_size *= -1


        means_dif = effect_size * control_std 
        test_mean = control_mean + means_dif 
        results['test_mean'].append(test_mean)
        results['means_dif'].append(means_dif)
        results['effect_size'].append(effect_size)
        results['total_sample_size'].append(total_sample_size)
        results['periods'].append(p)

    results['metric_type'] = metric_type
    results['alternative_hypothesis'] = alternative
    results['alpha'] = alpha
    results['power'] = power 
    results['control_mean'] = control_mean
    results['control_std'] = control_std
    results['subjects_per_period'] = subjects_per_period

    df = pd.DataFrame(results)

    # only include results within possible range 
    df = df.query("test_mean >= 0")
    if metric_type == 'proportion':
        df = df.query("test_mean <= 1")

    return df

st.title("A/B Test Designer")

with st.expander("About", expanded=True):
    st.write("""
        Use this app to help design and plan A/B tests. Start by entering the experiment details in the sidebar. 
        Then use the chart to see the minimum difference between the test and control group means the experiment would be
        able to detect with statistical significance after a given number of periods. Use this information to 
        understand how long the A/B test would need to run in order to provide the information required to make a decision. 
        Download the raw data for reference and sharing.   
    """)

with st.sidebar:
    subjects_per_period = st.number_input(
        "Subjects per Period", 
        min_value=1, 
        value=1000,
        help="""Number of unique subjects (users, devices, etc.) you expect to enter the experiment each period (day, week, etc.)."""
    )

    max_periods = st.number_input(
        "Max. Periods",
        min_value=1,
        value=14,
        help="""Maximum number of periods (days, weeks, etc.) you are willing to run the experiment."""
    )

    metric_type = st.selectbox(
        "Metric Type", 
        ["proportion", "mean"],
        help="""Select 'proportion' if the test's success metric is a conversion rate, and 'mean' if it's an average (e.g. revenue per user)."""
    )

    control_mean = st.number_input(
        "Metric Baseline", 
        min_value=0.0, 
        max_value=1.0 if metric_type == 'proportion' else None,
        value=0.1 if metric_type == 'proportion' else 10.0,
        help="""Expected value of the success metric for the control group."""
    )

    if metric_type == 'proportion':
        control_std = np.sqrt(control_mean * (1-control_mean))
    else:
        control_std = st.number_input(
            "Metric Standard Deviation", 
            min_value=0.0, 
            value=3.0,
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

tab1, tab2 = st.tabs(["Estimate Runtime", "Analyze Results"])

with tab1:
    results = get_min_detectable_difs(
        control_mean,
        control_std=control_std,
        max_periods=max_periods,
        subjects_per_period=subjects_per_period,
        metric_type=metric_type,
        alpha=alpha,
        power=power,
        alternative=alternative
    )

    fig = px.line(
        results,
        y='means_dif',
        x='periods',
        markers=True,
        title='Minimum Detectable Difference by Test Duration'
    )

    st.plotly_chart(fig)

    results[[
        'metric_type',
        'alternative_hypothesis',
        'alpha',
        'power',
        'control_mean',
        'test_mean',
        'means_dif',
        'control_std',
        'effect_size',
        'total_sample_size',
        'subjects_per_period',
        'periods'
    ]]
