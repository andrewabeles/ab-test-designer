import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
import math
from utils import RuntimeEstimator, TestResults

st.set_page_config(
    page_title="A/B Test Assistant",
    page_icon=':test_tube:',
    layout='wide'
)

st.title("A/B Test Assistant")

with st.expander("About", expanded=True):
    st.write("""
        This app assists in the design, planning, and analysis of A/B tests. 
        The Estimate Runtime tab helps determine the amount of time needed to run a future experiment, while the Analyze Results
        tab summarizes the outcome of a past or current one. 
        Start by entering the experiment parameters in the sidebar. Hover over an input widget's question mark icon to learn more
        about it. 
        The outputted data and visualizations on both tabs can be downloaded for reference and sharing.   
    """)

with st.sidebar:
    metric_type = st.selectbox(
        "Metric Type", 
        ["proportion", "mean"],
        help="""Select 'proportion' if the test's success metric is a conversion rate, and 'mean' if it's an average (e.g. revenue per user)."""
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
    n_groups = st.number_input(
        "Num. Groups",
        min_value=2,
        value=2,
        help="""Number of groups, or variants, you will test, including the control."""
    )
    bonferroni = st.checkbox(
        "Bonferroni Correction",
        help="""Divides alpha by the number of hypotheses being tested to reduce the risk of false positives. 
                Useful when testing multiple variants."""
    )
    if bonferroni:
        alpha = alpha / (n_groups - 1) # one hypothesis per non-control group

tab1, tab2 = st.tabs(["Estimate Runtime", "Analyze Results"])

with tab1:
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
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
        power = st.selectbox(
            "Power", 
            [0.8, 0.85, 0.9, 0.95],
            help="""1 - False Negative Rate. The probability the test will detect a minimum effect size with statistical significance if it truly exists."""
        )
    with col2:
        re = RuntimeEstimator(
            control_mean,
            control_std=control_std,
            n_groups=n_groups,
            max_periods=max_periods,
            subjects_per_period=subjects_per_period,
            metric_type=metric_type,
            alpha=alpha,
            power=power,
            alternative=alternative
        )
        fig = re.plot_results()
        st.plotly_chart(fig)

    with st.expander("Raw Data", expanded=False):
        st.write(re.print_results())

with tab2:
    uploaded_file = st.file_uploader(
        "Upload Test Results", 
        type='csv',
        help="""Upload a CSV file of your raw experiment results. Each row should represent an individual experiment subject 
        (user, device, etc.) and contain their observed metric value (binary conversion indicator, revenue, etc.) and the group 
        or variant to which they were assigned (test, control, etc.). File size is limited to 200MB."""
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1, col2, col3 = st.columns(3)
        with col1:
            y = st.selectbox(
                "Success Metric", 
                [c for c in df.select_dtypes(include=['number', 'bool']).columns],
                help="""Select the column containing the success metric values."""
            )
        with col2:
            group_id = st.selectbox(
                "Group Identifier", 
                [c for c in df.columns if c != y],
                help="""Select the column containing the group indicators."""
            )
        with col3:
            control_id = st.selectbox(
                "Control Group", 
                [i for i in df[group_id].unique()],
                help="""Select the control or reference group against which the other groups will be measured."""
            )
        test_results = TestResults(data=df, metric=y, group=group_id, control=control_id, metric_type=metric_type, alpha=alpha, alternative=alternative)
        if test_results.validate_metric_type():
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(test_results.plot_samples())
            with col2:
                st.plotly_chart(test_results.plot_differences())
            test_results_summary = test_results.summarize()
            test_results_summary
        else:
            st.warning(f"""The selected Success Metric '{y}' does not have the properties of the selected Metric Type '{metric_type}'.
                        Please update the Metric Type or select a different Success Metric column.""")