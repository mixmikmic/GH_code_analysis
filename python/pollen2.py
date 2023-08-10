from imports import *
# %mkdir cache
import joblib; mem = joblib.Memory(cachedir='cache')

get_ipython().magic('matplotlib inline')

from util.pollen_utils import pscale
import util.utils; reload(util.utils); from util.utils import (
    check_one2one, yrmths, flatten_multindex, ends_with,
    BatchArray, ravel, repackage_hidden, mse,
    replace_with_dummies,  filter_dtypes, log_,
    join_pollen_weather, read
)

date = lambda xs: dt.datetime(*xs)

dailydf = feather.read_dataframe('cache/dark_day.fth')

dailydf = (
    dailydf.sort_values('Time', ascending=True).reset_index(drop=1)
    .assign(Dt=lambda x: pd.to_datetime(x.Time, unit='s'))
    .assign(
        Day=lambda x: x.Dt.dt.day,
        Doy=lambda x: x.Dt.dt.dayofyear,
        M=lambda x: x.Dt.dt.month,
        Y=lambda x: x.Dt.dt.year,
        Day_int=lambda x: (x['Dt'] - x['Dt'].min()).dt.days,
    )
    .drop('Ozone', axis=1)  # This is a new field, I guess
)

dailydf[:3]

dailydf.loc[dailydf.eval('Precip_type != Precip_type'), 'Precip_type'] = 'none'
dailydf['Precip_accumulation'] = dailydf.Precip_accumulation.fillna(0)

def fill_pimt_null(s, timecol):
    """This column is null when there is no precipitation.
    Not sure of anything better to do, so I'm just setting
    it to the minimum time of the day in question
    """
    s2 = s.copy()
    null_ptime = s.isnull()
    s2.loc[null_ptime] = timecol[null_ptime]
    return s2.astype(int)

dailydf['Min_time'] = dailydf.Dt.map(lambda t: int(t.replace(hour=0).strftime('%s')))
dailydf.Precip_intensity_max_time = fill_pimt_null(dailydf.Precip_intensity_max_time, dailydf.Min_time)

from IPython.display import Image
Image('plots/cloud_cover_model_perf.png')

from sklearn.ensemble import RandomForestRegressor

def fill_cloud_cover_null(cc, X):
    """Solution wasn't obvious, so I just imputed the nulls
    with a random forest using the other columns.
    """
    null = cc != cc
    if not null.any():
        return cc

    rf = RandomForestRegressor(n_estimators=30, oob_score=True)
    rf.fit(X[~null], cc[~null])
    cc2 = cc.copy()
    cc2.loc[null] = rf.predict(X[null])
    return cc2


_feats = [k for k, d in dailydf.dtypes.items()
      if (d == float or d == int) and (k != 'Cloud_cover')
]
dailydf['Cloud_cover'] = fill_cloud_cover_null(dailydf.Cloud_cover, dailydf[_feats])

ddf = replace_with_dummies(dailydf, 'Icon Precip_type'.split())
assert (ddf == ddf).all().all(), "Don't want nulls here"

# Check that within a day the difference between maximum
# and minimum times are not greater than the
# number of seconds in a day

times = lfilter(lambda x: x.endswith('ime'), ddf)
minmax = DataFrame({
    'Min': ddf[times].min(axis=1),
    'Max': ddf[times].max(axis=1),
}).assign(Diff=lambda x: x.Max.sub(x.Min).div(60 * 60 * 24)) 

assert 0 <= minmax.Diff.max() <= 1, "All times within a day should be no more than 24 hrs apart"
minmax.Diff.max()  # should be no more than 1

assert (ddf[times].min(axis=1) == ddf.Min_time).all(), 'By definition'

unix_time_to_day_hrs = lambda s, min_time: (s - min_time) / 3600

for t in set(times) - {'Min_time'}:
    c = t + 's'
    ddf[c] = unix_time_to_day_hrs(ddf[t], ddf.Min_time)

slen = lambda x: len(set(x))
nunique = ddf.apply(slen)
ddf = ddf[nunique[nunique > 1].index].copy()

pscale

_, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))

cutoffs = np.array([1, 15, 90, 1500])
ax1.set_title('Linear scale')
ax1.plot(cutoffs)

ax2.set_title('Log scale')
ax2.plot(np.log10(cutoffs));

poldf = feather.read_dataframe('cache/pollen.fth')

xdf, xt, yt, rx, rxdf, ry = join_pollen_weather(
    poldf, ddf, time_cols=times, ycol='Logcnt'
)

print('|X|:', xt.size())
print('|y|:', yt.size())

print("Pollen count's 1st lag auto-correlation: {:.2%}"
      .format(xdf.Logcnt.corr(xdf.Logcnt.shift())))

# Sanity check that it's ordered ascending my date and not null
assert xdf.Dt.is_monotonic_increasing
assert xdf.Time.is_monotonic_increasing
assert (xdf.Doy > xdf.Doy.shift(1)).mean() > .98, (
    "Day of year int should increase once a year")

assert not xdf.isnull().any().any()

corrs = (rxdf.corrwith(ry).to_frame().rename(columns={0: 'Corr'})
         .assign(Abs=lambda x: x.Corr.abs())
         .sort_values('Abs', ascending=0).Corr)
assert corrs.abs().max() < .9
corrs[:5]

import torch as T
from torch.autograd import Variable
from torch import optim
from torch import nn

tofloat = lambda x: x.data[0]
unvar = lambda x: x.data.numpy().ravel()

class Rnn(nn.Module):
    def __init__(self, P=3, nhidden=21, num_layers=1, dropout=0):
        super().__init__()
        self.P, self.nhidden, self.num_layers, self.dropout = (
            P, nhidden, num_layers, dropout
        )
        self.rnn = nn.GRU(P, nhidden, num_layers, batch_first=True, dropout=dropout)
        self.Dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()
        self.zero_grad()

    def __dir__(self):
        return super().__dir__() + list(self._modules)
    
    def forward(self, input, hidden=None, outputh=False):
        if hidden is None:
            hidden = self.hidden
        out1, hout = self.rnn(input, hidden)
        out1d = self.Dropout(out1)
        out2 = self.decoder(ravel(out1d))
        self.hidden = repackage_hidden(hout)  # don't waste time tracking the grad
        if outputh:
            return out2, hout
        return out2
        
    def init_weights(self):
        initrange = 0.1
        for p in self.rnn.parameters():
            xavier_init(p.data)
        self.decoder.bias.data.fill_(0)
        xavier_init(self.decoder.weight.data)

    def init_hidden(self, bsz):
        "For lstm I'll need to return 2"
        weight = next(self.rnn.parameters()).data
        mkvar = lambda: Variable(weight.new(self.num_layers, bsz, self.nhidden).zero_())
        return mkvar()
    
    def set_hidden(self, bsz):
        h = self.init_hidden(bsz)
        self.hidden = h
        
        
def xavier_init(t):
    "This seems to be the recommended distribution for weight initialization"
    n = max(t.size())
    return t.normal_(std=n ** -.5)

criterion = nn.MSELoss()

def train_epoch(barray, model=None, hidden=None, optimizer=None, eval=False, batch_size=None):
    batch_size = batch_size or barray.batch_size
    assert batch_size or hidden
    hidden = model.init_hidden(batch_size) if hidden is None else hidden
    
    res = []
    ss, n = 0, 0

    for bix in barray.batch_ix_iter(batch_size=batch_size):
        x, y = barray[bix]
        optimizer.zero_grad()
        output = model(x, hidden)
        
        res.append(output.data.squeeze())
        if eval:
            continue

        loss = criterion(output, y.view(-1, 1))
        loss.backward()

        T.nn.utils.clip_grad_norm(model.parameters(), 3)
        optimizer.step()
        
        ss += tofloat(loss) * len(output)  # keep track of ss
        n += len(output)

    res = T.stack(res).view(-1).numpy()
    if eval:
        return res
    tot_loss = ss / n
    return tot_loss, res

def val_pred(model, warmup=True):
    if warmup:
        model.set_hidden(1)

    ix = int(not warmup)
    Dt = baval.Dt[ix]
    xs, ysv = baval[[ix]]

    ys = Series(unvar(ysv), index=Dt)
    yspred = model(xs)

    yspred_s = Series(unvar(yspred), index=Dt)
    return yspred_s, ys

# %mkdir /tmp/res/
VALFN = '/tmp/res/val.txt'
TRNFN = '/tmp/res/trn.txt'

def report_hook(model, res, vals=None):
    print()
    val_pred(model, warmup=True)
    yspred, ys = val_pred(model, warmup=False)
    val_acc = mse(yspred, ys)
    vals.append(val_acc)
    trn_acc = mse(ba.train_samples_y, res)

    with open(VALFN, 'a') as f:
        f.write('{:}\n'.format(val_acc))
    with open(TRNFN, 'a') as f:
        f.write('{:}\n'.format(trn_acc))
    print('{:,.3f}; val: {:,.4f}'.format(trn_acc, val_acc), end='; ')

    
def train_epochs(model, optimizer=None, rng=(500, ), print_every=10, report_hook=None, report_kw={}):
    with open(VALFN, 'w') as f: pass
    with open(TRNFN, 'w') as f: pass
    vals = []
    
    for i in range(*rng):
        _, res = train_epoch(ba, model=model, hidden=None, optimizer=optimizer)

        print('.', end='')
        if i % print_every == 0:
            if report_hook:
                report_hook(model, res, vals=vals)
        
    return res, min(vals)

# training batches
seq_len = 25
bcz = 32
ba = BatchArray(x=xt, y=yt, seq_len=seq_len, batch_size=bcz)

# validation batches
l = ba.num_leftover_rows
baval = BatchArray(x=xt[-2 * l:], y=yt[-2 * l:], seq_len=l, batch_size=1)

assert (xdf.index == rxdf.index).all(), 'Dropped some nulls?'
baval.Dt = [xdf.Dt.iloc[-2*l:-l], xdf.Dt.iloc[-l:]]
print('Training size: {}\nValidation size: {}'.format(ba.num_truncated_rows, l))

nhidden = 128
num_layers = 2
model = Rnn(P=rx.shape[-1], nhidden=nhidden, num_layers=num_layers, dropout=.05)
model.set_hidden(bcz)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
model

Image('plots/valid.png')

st = time.perf_counter()
res, mvals = train_epochs(model=model, optimizer=optimizer, rng=(25, ), print_every=10, report_hook=report_hook)
tt = time.perf_counter() - st

print('\n\nTime: {:.2f}'.format(tt))
print('Acc: {:.2f}; Val: {:.3f}'.format(mse(res, ba.train_samples_y), mvals))

(x_warm, y_warm) = baval[0]
(x_val, y_val) = baval[1]
y_val = y_val.data.numpy().ravel()

x_warm = x_warm.unsqueeze(0)
x_val = x_val.unsqueeze(0)

def eval_val(model, x_val):
    model(x_warm)
    val_pred = model(x_val).data  #.numpy().ravel()
    return val_pred

model.set_hidden(1)

get_ipython().run_cell_magic('time', '', '# ressv = np.array([eval_val(model, x_val) for _ in range(100)])\nressv = T.cat([eval_val(model, x_val) for _ in range(100)], 1).numpy()')

mu = ressv.mean(axis=1)
var = ressv.var(axis=1)

lparam = 50
tau = lparam**2 * (1 - model.dropout) / (2 * l * .9)
var += tau**-1

plt.figure(figsize=(16, 10))
dates = xdf.Dt[-l:].values
datify = lambda x: Series(x, index=dates)

datify(y_val).plot()
datify(mu).plot()

plt.legend(['Y', 'Pred'])

lo = datify(mu - var)
hi = datify(mu + var)
plt.fill_between(dates, lo, hi, alpha=.35, edgecolor='none')

resid = y_val - mu
diffy = y_val[1:] - y_val[:-1]
# diffpred = mu[1:] - mu[:-1]

plt.scatter(diffy, resid[1:], alpha=.3)
plt.xlabel('Daily difference')
plt.ylabel('Residual')
plt.text(-1, 1, 'Corr coef: {:.1%}'.format(np.corrcoef(diffy, resid[1:])[0][1]));

_, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 6))

sns.swarmplot(data=rxdf[-l:].assign(Resid=resid), x='Day_diff', y='Resid', ax=ax1)

# uncert_diff = (m9 - m10)[1:]
uncert_diff = var[1:]
ax2.scatter(uncert_diff, resid[1:], alpha=.3)
plt.xlabel('Daily difference')
plt.ylabel('Residual')
plt.text(.35, 1, 'Corr coef: {:.1%}'.format(np.corrcoef(uncert_diff, resid[1:])[0][1]));

