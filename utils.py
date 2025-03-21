import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 
import plotly.figure_factory as ff
from scipy import stats
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power

def get_min_detectable_difs(control_mean, control_std=None, n_groups=2, subjects_per_period=1000, max_periods=10, metric_type='proportion', alpha=0.05, power=0.8, alternative='two-sided'):
    results = {
        'test_mean': [],
        'means_dif': [],
        'effect_size': [],
        'total_sample_size': [],
        'periods': []
    }
    for p in range(1, max_periods+1):
        total_sample_size = p * subjects_per_period 
        sample_size_per_group = total_sample_size / n_groups
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

def plot_min_detectable_difs(min_detectable_difs):
    fig = px.line(
        min_detectable_difs,
        y='means_dif',
        x='periods',
        markers=True,
        title='Minimum Detectable Difference by Test Duration',
        labels={'means_dif': 'difference in means'}
    )
    return fig

class TestResults:
    def __init__(self, data=None, metric=None, group=None, control=None, metric_type='proportion', alpha=0.05, alternative='two-sided'):
        self.data = data 
        self.metric = metric 
        self.group = group 
        self.control = control 
        self.metric_type = metric_type 
        self.alpha = alpha
        self.alternative = alternative 
        samples = []
        for g in self.data[self.group].unique():
            x = self.data[self.data[self.group] == g][self.metric]
            sample = Sample(x, name=g, alpha=self.alpha, metric_type=self.metric_type)
            samples.append(sample)
        differences = pd.DataFrame()
        for i in samples:
            for j in samples:
                if i != j:
                    differences.loc[i.name, j.name] = i.test_difference(j, alpha=alpha, alternative=alternative) 
        self.samples = {s.name: s for s in samples}
        self.differences = differences     

    def validate_metric_type(self):
        for s in self.samples.values():
            if not s.validate_metric_type():
                return False
        return True
    
    def summarize(self):
        df = pd.DataFrame()
        for k, v in self.samples.items():
            df.loc[k, 'sample_size'] = v.n 
            df.loc[k, 'mean'] = v.mean 
            if k != self.control: 
                dif = self.differences[self.control][k]
                df.loc[k, 'difference'] = dif.difference 
                df.loc[k, 'lower_bound'] = dif.confidence_interval[0]
                df.loc[k, 'upper_bound'] = dif.confidence_interval[1]
                df.loc[k, 'p_value'] = round(dif.p_value, 4)
        return df     

    def plot_samples(self):
        if self.metric_type == 'proportion':
            fig = px.bar(
                x=[k for k in self.samples.keys()], # sample names 
                y=[v.mean for k, v in self.samples.items()], # sample proportions 
                error_y=[v.margin_of_error for k, v in self.samples.items()], # sample proportions' margins of error 
                labels={'x': 'group', 'y': 'mean'}
            )
        else:
            fig = px.histogram(
                x=self.data[self.metric],
                color=self.data[self.group], 
                barmode='overlay', 
                marginal='box'
            )
        fig.update_layout(title='Distribution by Group')
        return fig

    def plot_differences(self):
        difs = self.differences[self.control].dropna()
        error_x_plus = []
        error_x_minus = []
        for d in difs.values:
            if self.alternative == 'two-sided':
                error_x_plus.append(d.margin_of_error)
                error_x_minus.append(d.margin_of_error)
            elif self.alternative == 'larger':
                error_x_plus.append(0),
                error_x_minus.append(d.margin_of_error)
            else:
                error_x_plus.append(d.margin_of_error)
                error_x_minus.append(0)
        fig = px.scatter(
            x=[i.difference for i in difs.values],
            y=difs.index,
            error_x=error_x_plus,
            error_x_minus=error_x_minus,
            color=[i.statsig for i in difs.values],
            color_discrete_map={True: 'green', False: 'gray'},
            labels={'x': f'difference vs. {self.control}', 'y': 'group', 'color': 'stat. sig.'},
            title='Difference by Group'
        )    
        fig.add_vline(x=0, line_width=3, line_dash='dash', line_color='gray')
        return fig

class Sample:
    def __init__(self, x, name=None, alpha=0.05, metric_type='proportion'):
        if metric_type not in ['proportion', 'mean']:
            raise ValueError("Invalid metric type. Must be 'proportion' or 'mean'.")
        self.name = name 
        self.x = x
        self.alpha = alpha
        self.metric_type = metric_type
        self.n = len(x)
        self.n_unique = x.nunique()
        self.mean = x.mean()
        self.sum = x.sum() 
        self.var = np.var(x, ddof=1)
        if metric_type == 'proportion':
            self.critical_value = stats.norm.ppf(1 - alpha)
            self.standard_error = np.sqrt(self.mean * (1 - self.mean) / self.n)
        else:
            self.critical_value = stats.t.ppf(1 - alpha/2, df=self.n - 1)
            self.standard_error = x.std() / np.sqrt(self.n)
        self.margin_of_error = self.critical_value * self.standard_error
        self.confidence_interval = (self.mean - self.margin_of_error, self.mean + self.margin_of_error)
    
    def validate_metric_type(self):
        is_proportion_like = set(self.x.unique()).issubset({0, 1})
        if (self.metric_type == 'proportion' and not is_proportion_like) or (self.metric_type == 'mean' and is_proportion_like):
            return False
        else:
            return True
    
    def test_difference(self, control_sample, alpha=0.05, alternative='two-sided'):
        return SampleDifference(self, control_sample, alpha=alpha, alternative=alternative)

class SampleDifference:
    def __init__(self, sample_test, sample_control, alpha=0.05, alternative='two-sided'):
        if alternative not in ['two-sided', 'larger', 'smaller']:
            raise ValueError("Invalid alternative hypothesis. Must be 'two-sided', 'larger', or 'smaller'.")
        if sample_test.metric_type != sample_control.metric_type:
            raise ValueError("Cannot compare samples with different metric types.")
        self.metric_type = sample_test.metric_type
        self.sample_test = sample_test
        self.sample_control = sample_control 
        self.n = sample_test.n + sample_control.n
        self.alpha = alpha
        self.alternative = alternative
        self.difference = sample_test.mean - sample_control.mean
        auc = 1 - alpha/2 if alternative == 'two-sided' else 1 - alpha
        if self.metric_type == 'proportion':
            self.statistic, self.p_value = proportions_ztest([sample_test.sum, sample_control.sum], [sample_test.n, sample_control.n], alternative=self.alternative)
            p_pooled = (sample_test.sum + sample_control.sum) / (sample_test.n + sample_control.n)
            self.standard_error = np.sqrt(p_pooled * (1 - p_pooled) * (1/sample_test.n + 1/sample_control.n))
            self.critical_value = stats.norm.ppf(auc)
        else:
            self.statistic, self.p_value, self.dof = ttest_ind(sample_test.x, sample_control.x, alternative=self.alternative)
            self.standard_error = np.sqrt(sample_test.var/sample_test.n + sample_control.var/sample_control.n)
            self.critical_value = stats.t.ppf(auc, df=self.dof)
        self.statsig = self.p_value < alpha
        self.margin_of_error = self.critical_value * self.standard_error
        if alternative == 'two-sided':
            self.confidence_interval = (self.difference - self.margin_of_error, self.difference + self.margin_of_error)
        elif alternative == 'larger':
            self.confidence_interval = (self.difference - self.margin_of_error, np.inf)
        else:
            self.confidence_interval = (-np.inf, self.difference + self.margin_of_error)
