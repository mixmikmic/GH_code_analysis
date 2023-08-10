get_ipython().magic('matplotlib inline')
from decimal import Decimal
import itertools
from pprint import pprint
import math

import pandas
import seaborn
from matplotlib import pyplot as plt
plt.style.use('seaborn-poster')

import metrics
import simulate
import harvesting
import market
import plot
import mortality
from plot import plt
import withdrawal
from portfolio import Portfolio
import montecarlo

get_ipython().magic('pdb on')

series = market.Returns_US_1871()
START_YEAR=1871

#series = market.Japan_1957()
#START_YEAR=1957

series = market.UK1900()
START_YEAR = 1900

x = simulate.withdrawals(series.iter_from(1966),
                                 withdraw=lambda p, s: withdrawal.LonginvestSmoothing(withdrawal.VPW(p, s, years_left=50)),
                                 years=30,
                                 portfolio=(1000000, 0),
                                 harvesting=harvesting.make_rebalancer(1))
#[float(n.withdraw_n) for n in x]

def test_all(year, strategies_to_test, years=25):
    results = {}
    for h in strategies_to_test:
        starting_portfolio = (1000000,0)

        # siamond used VPW with a 50 year depletion period, so I'll try that as well
        x = simulate.withdrawals(series.iter_from(year),
                                 withdraw=lambda p, s: withdrawal.VPW(p, s, years_left=35),
                                 #withdraw=lambda p, s: withdrawal.ConstantDollar(p, s, rate=Decimal('.026')),
                                 #withdraw=lambda p, s: withdrawal.SteinerSmoothing(withdrawal.VPW(p, s, years_left=50)),
                                 #withdraw=lambda p, s: withdrawal.ECM(p, s),
                                 years=years,
                                 portfolio=starting_portfolio,
                                 harvesting=h)
        results[h.__name__] = x
    return results

def make_data_tables(strategies_to_test, lens, years=25):
    frames = {}
    
    for s in strategies_to_test:
        frames[s.__name__] = pandas.DataFrame(columns=['Y%02d' % y for y in range(years)])
        
    last_year = 2015 - years
    
    for i in range(START_YEAR, last_year):
        n = test_all(i, strategies_to_test, years=years)
        for s in n.keys():
            frames[s].loc[i] = [lens(_) for _ in n[s]]

    return frames

def semideviation(frame):
    #goal = frame.mean()
    goal = 40000
    values = frame[lambda s: s < goal]
    sumvalues = sum(((goal - v) ** 2 for v in values))
    average = sumvalues / len(values)
    return math.sqrt(average)


def calculate_stuff(df, use_all_columns=True):
    """
    Things we don't yet calculate.
    Real returns (that an investor sees)
    Average asset allocation of bonds
    Minimum asset allocation of bond
    """
    
    if use_all_columns:
        stack = df.stack()
    else:
        columns = df.columns.tolist()
        stack = df[columns[-1]]
    
    return ({
            'Mean': round(stack.mean()),
            'Median' : round(stack.median()),
            'Stddev': stack.std() / stack.mean(),
            'Min': stack.min(),
            'Max' : stack.max(),
            '0.1h percentile' : round(stack.quantile(.001)),        
            ' 1st percentile' : round(stack.quantile(.01)),
            ' 5th percentile' : round(stack.quantile(.05)),
            '10th percentile' : round(stack.quantile(.1)),
            '90th percentile' : round(stack.quantile(.9)),
            'Mean of 25% lowest' : round(stack.nsmallest(int(len(stack) / 4)).mean()),
            'Kurtosis' : stack.kurtosis(),
            'Skew' : stack.skew(),
            'Semidev-4' : semideviation(stack),
    })

def make_all_stats():
    strategies_to_test = [
#        harvesting.N_35_RebalanceHarvesting,
        harvesting.N_60_RebalanceHarvesting,
#        harvesting.N_100_RebalanceHarvesting,
#        harvesting.BondsFirst,
#        harvesting.AltPrimeHarvesting,
#        harvesting.PrimeHarvesting,
#        harvesting.AgeBased_100,
#        harvesting.AgeBased_110,
#        harvesting.AgeBased_120,
#        harvesting.Glidepath,
#        harvesting.OmegaNot,
#        harvesting.Weiss,
#        harvesting.ActuarialHarvesting,
    ]

    #t = make_data_tables(strategies_to_test, lambda x: round(x.portfolio_r), years=40)
    t = make_data_tables(strategies_to_test, lambda x: round(x.withdraw_r), years=30)
    #t = make_data_tables(strategies_to_test, lambda x: round(x.portfolio_bonds/x.portfolio_n*100), years=30)

    if False:
        fn_mort = mortality.make_mortality_rate()
        for key in t:
            age = 65
            for c in t[key].columns:
                t[key][c] *= (1 - fn_mort(age, mortality.FEMALE))
                age += 1


    if False:
        for k in t:
            t[k].to_csv('CSV - withdraw - %s.csv' % k)
            

    df = None

    for key in sorted(t.keys()):
        #t[key].to_csv('CSV - portfolio - %s.csv' % key)
        stats = calculate_stuff(t[key], use_all_columns=True)

        if df is None:
            # We need to know the columns in order to define a data frame,
            # so we defer the creation until now
            df = pandas.DataFrame(columns=sorted(list(stats.keys())))
        df.loc[key] = stats
        
        seaborn.distplot(t[key].stack(), label=key, axlabel='Annual Withdrawal ($$$)')
        plt.legend(loc=0)

    return df

d = make_all_stats()
#d.to_csv('CSV-comparison.csv')
d

def compare_year_lens(year, lens, title):
    strategies_to_test = [
        harvesting.N_60_RebalanceHarvesting,
        harvesting.N_100_RebalanceHarvesting,
#        harvesting.N_0_RebalanceHarvesting,
#        harvesting.N_35_RebalanceHarvesting,
#        harvesting.OmegaNot,
#        harvesting.Weiss,
#        harvesting.BondsFirst,
#        harvesting.AltPrimeHarvesting,
#        harvesting.PrimeHarvesting,
#        harvesting.ActuarialHarvesting,
    ]

    results = test_all(year, strategies_to_test, years=30)
    
    fig, ax = plt.subplots()

    if '%' not in title:
        plot.format_axis_labels_with_commas(ax.get_yaxis())

    plt.xlabel('Year of Retirement')
    plt.title('Retiring in %s (%s)' % (year, title))

    for strategy in (sorted(results.keys())):
        ax_n = fig.add_subplot(111, sharex=ax, sharey=ax)
        ws = [lens(n) for n in results[strategy]]
        ax_n.plot(ws, label=strategy)
        ax_n.set_ymargin(0.05)
    plt.legend(loc=0)
    ax.set_ylim(bottom=0)
    plt.show()
    
    s = {}
    for strategy in results.keys():
        ws = [lens(n) for n in results[strategy]]
        s[strategy] = ws
    df = pandas.DataFrame(data=s)
    diff = (df['60% Stocks'] - df['100% Stocks'])
    print(diff.sum())
    print(diff.loc[lambda x: x > 0].mean())
    print(diff)

def chart_all(year):
#    compare_year_lens(year, lambda x: x.portfolio_stocks/x.portfolio_n*100, "Stock %")
#    compare_year_lens(year, lambda x: x.portfolio_n, "Portfolio (Nominal)")
#    compare_year_lens(year, lambda x: x.portfolio_r, "Portfolio (Real)")
#    compare_year_lens(year, lambda x: x.withdraw_n, "Withdrawals (Nominal)")
    compare_year_lens(year, lambda x: x.withdraw_r, "Withdrawals (Real)")
    
chart_all(1900)

survival_fn = mortality.make_mortality(mortality.ANNUITY_2000)

def get_rq(portfolio, age, withdrawal_pct):
    # I can't figure out how to to joint life expectancy so I'll
    # just use female life expectancy for now :/
    life_expectancy = mortality.life_expectancy(None, age)

    stock_pct = round(portfolio.stocks_pct * 100)
    mean = montecarlo.simba_mean[stock_pct]
    stddev = montecarlo.simba_stddev[stock_pct]
    
    return metrics.probability_of_ruin(mean, stddev, life_expectancy, float(withdrawal_pct))

def simulate_risk_quotient(series,
                            portfolio=(600000, 400000),
                            harvesting=harvesting.PrimeHarvesting,
                            withdraw=withdrawal.VPW,
                            live_until=None):
    portfolio = Portfolio(portfolio[0], portfolio[1])
    strategy = harvesting(portfolio).harvest()
    strategy.send(None)
    withdrawal_strategy = withdraw(portfolio, strategy).withdrawals()
    annual = []

    age = 65
    if not live_until:
        live_until = mortality.gen_age(survival_fn)

    # Withdrawals happen at the start of the year, so the first time
    # we don't have any performance data to send them....
    data = withdrawal_strategy.send(None)
    # Every year after the withdrawal we recalculate our risk quotient.
    rq = get_rq(portfolio, age, data.withdraw_n/data.portfolio_n)
    annual.append(rq)

    for d in series:
        age += 1
        if age > live_until:
            break

        data = withdrawal_strategy.send(d)
        rq = get_rq(portfolio, age, data.withdraw_n/data.portfolio_n)
        annual.append(rq)
    return annual

def compare_year(year):
    strategies_to_test = [
#        harvesting.N_30_RebalanceHarvesting,
#        harvesting.N_40_RebalanceHarvesting,
#        harvesting.N_50_RebalanceHarvesting,
        harvesting.make_rebalancer(.6),
#        harvesting.AltPrimeHarvesting,
#        harvesting.PrimeHarvesting,
    ]

    results = test_all(year, strategies_to_test, years=30)['Rebalancer']
    rqs = simulate_risk_quotient(series.iter_from(year), harvesting=harvesting.make_rebalancer(.6), live_until=95, withdraw=lambda p, s: withdrawal.VPW(p, s, years_left=50))
    
    def lens(x):
        return x.withdraw_r
    
    fig, ax = plt.subplots()

    plt.xlabel('Year of Retirement')
    plt.title('Retirement in Year %s' % year)

#    ax_n = fig.add_subplot(111, sharex=ax, sharey=ax)
    ws = [lens(n) for n in results]
    ax.plot(ws, label='Withdrawals', color='g')
#    ax.set_ymargin(0.05)
    ax.set_ylim(bottom=0)
    
    ax2 = ax.twinx()
    ax2.plot(rqs, label='Risk Quotient', color='r')
    ax2.set_ylim(bottom=0)

    plt.legend(loc=0)
    plt.show()
    
compare_year(1966)



