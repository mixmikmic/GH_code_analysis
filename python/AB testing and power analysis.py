get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=1.5)
sns.set_style('darkgrid') 
plt.rcParams['figure.figsize'] = 12, 8
import scipy
from scipy.stats import binom 
N = 100
ps = [0.03, 0.3, 0.6, 0.97]
for idx, p in enumerate(ps):
    x = scipy.linspace(0, N, N + 1)
    plt.plot(x, binom.pmf(x, N, p))
    text_x = (N * p) - 1
    text_y = 0.1
    plt.text(text_x, text_y + 0.012, 'p = {:.2f}'.format(p))
    plt.text(text_x, text_y, 'N*p = {:.0f}'.format(N * p))
    plt.text(text_x, text_y - 0.012, 'N*(1-p) = {:.0f}'.format(N * (1-p)))
plt.show()

from scipy.stats import norm
one_tailed_confidence_interval = 0.975
z_score = norm.ppf(one_tailed_confidence_interval)  # Percent Point Function (Inverse of CDF)
print('z-score for two-tailed 95% confidence interval: {:.2f}'.format(z_score))

import numpy as np
x_axis = np.arange(-3, 3, 0.01)
plt.plot(x_axis, norm.pdf(x_axis,0,1))
x_axis_fill = np.arange(-z_score, z_score, 0.01)
plt.fill_between(x_axis_fill, norm.pdf(x_axis_fill,0,1))
text_y = 0.2
plt.axvline(z_score, linestyle='dashed')
plt.axvline(-z_score, linestyle='dashed')
plt.text(-0.3, text_y, '95%', color='white', size=30)
plt.text(z_score + 0.05, text_y, '{:0.2f}'.format(z_score), size=30, rotation=90)
plt.text(- z_score + 0.05, text_y, '-{:0.2f}'.format(z_score), size=30, rotation=90)
plt.title('Two-tailed 95% confidence interval of the z-distribution ($\mu = 0$, $\sigma = 1$)')
plt.show()

one_tailed_confidence_interval = 0.995
z_score = norm.ppf(one_tailed_confidence_interval)
print('z-score for two-tailed 99% confidence interval: {:.2f}'.format(z_score))

X = 300
N = 2000
p = X / N
SE = (p * (1-p) / N)**0.5
margin_of_error = z_score * SE

step_size = 0.001
x_axis = np.arange(p - 3*margin_of_error, p + 3*margin_of_error + step_size, step_size)
plt.plot(x_axis, norm.pdf(x_axis, p, SE))
x_axis_fill = np.arange(p - margin_of_error, p + margin_of_error, step_size)
plt.fill_between(x_axis_fill, norm.pdf(x_axis_fill, p, SE))
text_y = 20
plt.axvline(p + margin_of_error, linestyle='dashed')
plt.axvline(p - margin_of_error, linestyle='dashed')
plt.text(p - 0.005, text_y, '99%', color='white', size=30)
plt.text(p + margin_of_error + 0.001, text_y, '{:0.3f}'.format(p + margin_of_error), size=30, rotation=90)
plt.text(p - margin_of_error + 0.001, text_y, '{:0.3f}'.format(p - margin_of_error), size=30, rotation=90)
plt.title('Margin of error for 99% confidence interval')
plt.show()

def calc_p(X_hyp, N_hyp):
    return X_hyp / N_hyp
    
def calc_SE(p_hyp, N_hyp):
    return (p_hyp * (1-p_hyp) / N_hyp)**0.5
    
def plot_hypothesis(p_hyp, SE_hyp, lbl_txt):
    step_size = 0.001
    x_axis = np.arange(0.12, 0.225 + step_size, step_size)
    plt.plot(x_axis, norm.pdf(x_axis, p_hyp, SE_hyp), label=lbl_txt)

X_ctrl, N_ctrl = 300, 2000
X_exp, N_exp = 400, 2200
plot_hypothesis(calc_p(X_ctrl, N_ctrl), calc_SE(calc_p(X_ctrl, N_ctrl), N_ctrl), lbl_txt='Control')
plot_hypothesis(calc_p(X_exp, N_exp), calc_SE(calc_p(X_exp, N_exp), N_exp), lbl_txt='Experiment')

p_pool = (X_ctrl + X_exp) / (N_ctrl + N_exp)
SE_pool = (p_pool*(1-p_pool)*((1/N_ctrl) + (1/N_exp)))**0.5
plot_hypothesis(p_pool, SE_pool, lbl_txt='Pooled, $H_{0}$')
plt.legend(loc='best')
plt.show()

X_ctrl, N_ctrl, X_exp, N_exp = 900, 10000, 1200, 9000
p_pool = (X_ctrl + X_exp) / (N_ctrl + N_exp)
print('p_pool is {:.4f}'.format(p_pool))
SE_pool = (p_pool * (1-p_pool) *((1/N_ctrl)+(1/N_exp)))**0.5
print('SE_pool is {:.4f}'.format(SE_pool))
d = (X_exp / N_exp) - (X_ctrl / N_ctrl)
print('d hat, the estimated difference, is {:.4f}'.format(d))
one_tailed_confidence_interval = 0.975
z_score_95 = norm.ppf(one_tailed_confidence_interval)
margin = z_score_95 * SE_pool
print('margin of error is {:.4f}'.format(margin))
print('the confidence interval spans {:.4f} to {:.4f}'.format(d - margin, d + margin))

one_tailed_confidence_interval = 0.975
z_score = norm.ppf(one_tailed_confidence_interval)
print('z-score for two-tailed 95% confidence interval: {:.2f}'.format(z_score))

def plot_marg_of_error(X, N, text_y, title):
    p = X / N
    SE = (p * (1-p) / N)**0.5
    margin_of_error = z_score * SE

    step_size = 0.001
    x_axis = np.arange(0.33, 0.65, step_size)
    plt.plot(x_axis, norm.pdf(x_axis, p, SE))
    x_axis_fill = np.arange(p - margin_of_error, p + margin_of_error, step_size)
    plt.fill_between(x_axis_fill, norm.pdf(x_axis_fill, p, SE))
    plt.axvline(p + margin_of_error, linestyle='dashed')
    plt.axvline(p - margin_of_error, linestyle='dashed')
    plt.text(p - 0.015, text_y, '95%', color='white', size=30)
    plt.text(p + margin_of_error + 0.001, text_y, '{:0.3f}'.format(p + margin_of_error), size=30, rotation=90)
    plt.text(p - margin_of_error + 0.001, text_y, '{:0.3f}'.format(p - margin_of_error), size=30, rotation=90)
    plt.title(title)

X = 50
N = 100
plt.subplot('211')
plot_marg_of_error(X, N, 5, '100 samples')
X = 500
N = 1000
plt.subplot('212')
plot_marg_of_error(X, N, 15, '1000 samples')
plt.subplots_adjust(hspace=0.4)
plt.show()

# First let's define a function for plotting the null and alternative hypotheses
def plot_null_and_alternative_hypotheses(X_ctrl, N_ctrl, X_exp, N_exp, title):
    one_tailed_confidence_interval = 0.975
    z_score = norm.ppf(one_tailed_confidence_interval)
    print('z-score for two-tailed 95% confidence interval: {:.2f}'.format(z_score))

    step_size = 0.001
    x_min, x_max = 0.35, 0.55
    x_axis = np.arange(x_min, x_max, step_size)

    # The null hypothesis
    p_null = (X_ctrl + X_exp) / (N_ctrl + N_exp)
    SE_null = (p_null*(1-p_null)*((1/N_ctrl) + (1/N_exp)))**0.5
    margin_of_error_null = z_score * SE_null
    plt.plot(x_axis, norm.pdf(x_axis, p_null, SE_null), label='$H_{0}$')

    # SE = SD / sqrt(N)
    cohens_d = ((X_exp / N_exp) - (X_ctrl / N_ctrl))/np.sqrt(p_null*(1-p_null))
    print("cohen's d is {}".format(cohens_d))
    
    # Type I error, false positive rate if the null hypothesis is true
    x_axis_fill = np.arange(p_null + margin_of_error_null, x_max, step_size)
    plt.fill_between(x_axis_fill, norm.pdf(x_axis_fill, p_null, SE_null), facecolor='blue', alpha=0.5, label='Type I error')

    # The alternative hypothesis
    X_alt = X_exp
    N_alt = N_exp
    p_alt = calc_p(X_alt, N_alt)
    SE_alt = calc_SE(p_alt, N_alt)
    plt.plot(x_axis, norm.pdf(x_axis, p_alt, SE_alt), label='$H_{A}$')

    # Type II error, false positive rate if the null hypothesis is true
    x_axis_fill = np.arange(x_min, p_null + margin_of_error_null, step_size)
    plt.fill_between(x_axis_fill, norm.pdf(x_axis_fill, p_alt, SE_alt), facecolor='orange', alpha=0.5, label='Type II error')
    plt.title(title)
    plt.legend(loc='best')

X_ctrl, N_ctrl = 400, 1000
X_exp, N_exp = 450, 1000
plot_null_and_alternative_hypotheses(X_ctrl, N_ctrl, X_exp, N_exp, title='2000 samples')
plt.show()

X_ctrl, N_ctrl = 400*7, 1000*7
X_exp, N_exp = 450*7, 1000*7
plot_null_and_alternative_hypotheses(X_ctrl, N_ctrl, X_exp, N_exp, title='Increase N to 14000 samples')
plt.show()

X_ctrl, N_ctrl = 320, 1000
X_exp, N_exp = 500, 1000
plot_null_and_alternative_hypotheses(X_ctrl, N_ctrl, X_exp, N_exp, title="Larger effect size, Type II errors vanish")
plt.show()

z_score_975 = norm.ppf(0.975)
print('z-score for two-tailed 95% confidence interval: {:.2f}'.format(z_score_975))
z_score_80 = norm.ppf(0.8)
print('z-score for 80% power (20% Type II error rate): {:.2f}'.format(z_score_80))

X_ctrl, N_ctrl, X_exp, N_exp = 900, 10000, 1200, 9000
p_ctrl = calc_p(X_ctrl, N_ctrl)
p_exp = calc_p(X_exp, N_exp)
p_pool = (X_ctrl + X_exp) / (N_ctrl + N_exp)
d = p_exp - p_ctrl
SD = np.sqrt(p_pool*(1-p_pool))
ES = d / SD
2 * ((z_score_975 + z_score_80) / ES)**2

def convert_to_power(N, d, p_pool):
    ES = d / np.sqrt(p_pool * (1-p_pool))
    z_score_80 = (np.sqrt(N / 2) * ES) - norm.ppf(0.975)
    return norm.cdf(z_score_80)
convert_to_power(821, d, p_pool)

x_axis = np.arange(0, 4000, 1)
plt.plot(x_axis, convert_to_power(x_axis, d, p_pool))
plt.axvline(800, linestyle='dashed')
plt.xlabel('sample size')
plt.ylabel('power')
plt.show()

x_axis = np.arange(0, 0.15, 0.001)
plt.plot(x_axis, convert_to_power(800, x_axis, p_pool))
plt.xlabel('difference in means')
plt.ylabel('power')
plt.show()

x_axis = np.arange(0, 1, 0.001)
plt.plot(x_axis, convert_to_power(800, d, x_axis))
plt.xlabel('pooled p')
plt.ylabel('power')
plt.show()

# Reminds of the plots here https://www.statisticsdonewrong.com/power.html



