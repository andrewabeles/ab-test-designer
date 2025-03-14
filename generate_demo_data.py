import numpy as np
import pandas as pd 

n = 1000 # sample size per group
d = 0.2 # percentage increase of test over control
mean_ctrl = 10 # control group mean 
std = 2 # control group standard deviation 
p_ctrl = 0.2 # control group proportion 

def generate_data(mean, std=None, d=0.1, n=100, metric_type='mean'):
    mean_test = mean * (1 + d)
    df = pd.DataFrame()
    if metric_type == 'mean':
        df['control'] = np.random.normal(loc=mean, scale=std, size=n)
        df['test'] = np.random.normal(loc=mean_test, scale=std, size=n)
    elif metric_type == 'proportion':
        df['control'] = np.random.binomial(n=1, p=mean, size=n)
        df['test'] = np.random.binomial(n=1, p=mean_test, size=n)
    df_melted = pd.melt(df, var_name='group')
    return df_melted 

def main():
    df_mean = generate_data(mean_ctrl, std=std, d=d, n=n, metric_type='mean')
    df_prop = generate_data(p_ctrl, std=None, d=d, n=n, metric_type='proportion')
    df_mean.to_csv('data/demo_mean.csv', index=False)
    df_prop.to_csv('data/demo_prop.csv', index=False)

if __name__ == '__main__':
    main()
