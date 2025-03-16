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
    if metric_type not in ['proportion', 'mean']:
        raise ValueError("Invalid metric type. Must be 'proportion' or 'mean'.")
    if alternative not in ['two-sided', 'larger', 'smaller']:
        raise ValueError("Invalid alternative hypothesis. Must be 'two-sided', 'larger', or 'smaller'.")

    values_test = df[df[group] == 'test'][metric]
    values_ctrl = df[df[group] == 'control'][metric]
    n_test = len(values_test)
    n_ctrl = len(values_ctrl)
    dof = n_test + n_ctrl - 2 
    sum_test = values_test.sum()
    sum_ctrl = values_ctrl.sum()
    mean_test = values_test.mean()
    mean_ctrl = values_ctrl.mean()
    var_test = np.var(values_test, ddof=1)
    var_ctrl = np.var(values_ctrl, ddof=1)
    diff = mean_test - mean_ctrl 

    if metric_type == 'mean':
        # T-Test for two means 
        stat, p_val = ttest_ind(values_test, values_ctrl, alternative=alternative)
        # Compute standard error 
        se = np.sqrt((var_test/n_test) + (var_ctrl/n_ctrl))
        # Get critical value 
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, df=dof)
        else:
            critical_value = stats.t.ppf(1 - alpha, df=dof)

    elif metric_type == 'proportion':
        # Z-test for two proportions 
        count = np.array([sum_test, sum_ctrl])
        nobs = np.array([n_test, n_ctrl])
        stat, p_val = proportions_ztest(count, nobs, alternative=alternative)
        # Compute standard error 
        p_pool = (sum_test + sum_ctrl) / (n_test + n_ctrl)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_test + 1 / n_ctrl))
        # Get critical value
        if alternative == 'two-sided':
            critical_value = stats.norm.ppf(1 - alpha/2)
        else:
            critical_value = stats.norm.ppf(1 - alpha)

    # Compute confidence interval 
    if alternative == 'two-sided':
        ci_low, ci_high = diff - critical_value * se, diff + critical_value * se
    elif alternative == 'larger':
        ci_low, ci_high = diff - critical_value * se, np.inf 
    elif alternative == 'smaller':
        ci_low, ci_high = -np.inf, diff + critical_value * se

    return {
        'control': mean_ctrl,
        'test': mean_test,
        'difference': diff,
        'statistic': stat,
        'p_value': p_val,
        'confidence_interval': (ci_low, ci_high)
    }
