import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
from utils import get_min_detectable_difs

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

with tab2:
    uploaded_file = st.file_uploader("Upload Test Results", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            y = st.selectbox("Success Metric", [c for c in df.columns])
        with col2:
            group_id = st.selectbox("Group Identifier", [c for c in df.columns])

    