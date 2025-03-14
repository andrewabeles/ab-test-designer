import pandas as pd 
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power

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