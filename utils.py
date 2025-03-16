import pandas as pd 
import numpy as np
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

def get_test_results(df, group, metric, metric_type='proportion', alpha=0.05, alternative='two-sided'):
    results = {}
    samples = []
    for g in df[group].unique():
        x = df[df[group] == g][metric]
        sample = Sample(x, name=g, alpha=alpha, metric_type=metric_type)
        samples.append(sample)
    differences = pd.DataFrame()
    for i in samples:
        for j in samples:
            if i != j:
                differences.loc[i.name, j.name] = i.test_difference(j, alpha=alpha, alternative=alternative) 
    results['samples'] = {s.name: s for s in samples}
    results['differences'] = differences
    return results

class Sample:
    def __init__(self, x, name=None, alpha=0.05, metric_type='proportion'):
        if metric_type not in ['proportion', 'mean']:
            raise ValueError("Invalid metric type. Must be 'proportion' or 'mean'.")
        self.name = name 
        self.x = x
        self.alpha = alpha
        self.metric_type = metric_type
        self.n = len(x)
        self.mean = x.mean()
        self.sum = x.sum() 
        if metric_type == 'proportion':
            self.critical_value = stats.norm.ppf(1 - alpha)
            self.standard_error = np.sqrt(self.mean * (1 - self.mean) / self.n)
        else:
            self.critical_value = stats.t.ppf(1 - alpha/2, df=self.n - 1)
            self.standard_error = x.std() / np.sqrt(self.n)
        self.margin_of_error = self.critical_value * self.standard_error
        self.confidence_interval = (self.mean - self.margin_of_error, self.mean + self.margin_of_error)

    def test_difference(self, control_sample, alpha=0.05, alternative='two-sided'):
        dif = SampleDifference(self, control_sample, alpha=alpha, alternative=alternative)
        return dif

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
        if self.metric_type == 'proportion':
            self.statistic, self.p_value = proportions_ztest([sample_test.sum, sample_control.sum], [sample_test.n, sample_control.n], alternative=self.alternative)
            self.standard_error = np.sqrt(sample_test.mean * (1 - sample_test.mean) / sample_test.n + sample_control.mean * (1 - sample_control.mean) / sample_control.n)
            self.critical_value = stats.norm.ppf(1 - alpha)
        else:
            self.statistic, self.p_value, self.dof = ttest_ind(sample_test.x, sample_control.x, alternative=self.alternative)
            self.standard_error = np.sqrt(sample_test.standard_error**2 / sample_test.n + sample_control.standard_error**2 / sample_control.n)
            self.critical_value = stats.t.ppf(1 - alpha/2, df=self.dof)
        self.margin_of_error = self.critical_value * self.standard_error
        if self.alternative == 'two-sided':
            self.confidence_interval = (self.difference - self.margin_of_error, self.difference + self.margin_of_error)
        elif self.alternative == 'larger':
            self.confidence_interval = (self.difference - self.margin_of_error, np.inf)
        else:
            self.confidence_interval = (-np.inf, self.difference + self.margin_of_error)
