# manage data and fit
import pandas as pd
import numpy as np

# first part with least squares
from scipy.optimize import curve_fit

# second part about ODR
from scipy.odr import ODR, Model, Data, RealData

# style and notebook integration of the plots
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style="whitegrid")

df = pd.read_csv("donnees_exo9.csv", sep=";")
df.head(8) # first 8 lines

ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)

def f_model(x, a, c):
    return pd.np.log((a + x)**2 / (x - c)**2)

df["model"] = f_model(df["x"], 3, -2)
df.head(8)

ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)
ax = df.plot(
    x="x", y="model",
    kind="line", ax=ax, linewidth=1
)

help(curve_fit)

popt, pcov = curve_fit(
    f=f_model,       # model function
    xdata=df["x"],   # x data
    ydata=df["y"],   # y data
    p0=(3, -2),      # initial value of the parameters
    sigma=df["Dy"]   # uncertainties on y
)

print(popt)

a_opt, c_opt = popt
print("a = ", a_opt)
print("c = ", c_opt)

perr = np.sqrt(np.diag(pcov))
Da, Dc = perr
print("a = %6.2f +/- %4.2f" % (a_opt, Da))
print("c = %6.2f +/- %4.2f" % (c_opt, Dc))

R2 = np.sum((f_model(df.x, a_opt, c_opt) - df.y.mean())**2) / np.sum((df.y - df.y.mean())**2)
print("r^2 = %10.6f" % R2)

df["model"] = f_model(df.x, a_opt, c_opt)
df.head()

ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)
ax = df.plot(
    x="x", y="model",
    kind="line", ax=ax, linewidth=1
)

x = np.linspace(0, 20, 200)
ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)
ax.plot(x, f_model(x, a_opt, c_opt), linewidth=1)

nval = len(df)
del df["model"]
Dx = [np.random.normal(0.3, 0.2) for i in range(nval)]
df["Dx"] = Dx
df.head()

ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", xerr="Dx",
    title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)

def fxy_model(beta, x):
    a, c = beta
    return pd.np.log((a + x)**2 / (x - c)**2)

data = RealData(df.x, df.y, df.Dx, df.Dy)
model = Model(fxy_model)

odr = ODR(data, model, [3, -2])
odr.set_job(fit_type=2)
lsq_output = odr.run()
print("Iteration 1:")
print("------------")
print("   stop reason:", lsq_output.stopreason)
print("        params:", lsq_output.beta)
print("          info:", lsq_output.info)
print("       sd_beta:", lsq_output.sd_beta)
print("sqrt(diag(cov):", np.sqrt(np.diag(lsq_output.cov_beta)))

# if convergence is not reached, run again the algorithm
if lsq_output.info != 1:
    print("\nRestart ODR till convergence is reached")
    i = 1
    while lsq_output.info != 1 and i < 100:
        print("restart", i)
        lsq_output = odr.restart()
        i += 1
    print("   stop reason:", lsq_output.stopreason)
    print("        params:", lsq_output.beta)
    print("          info:", lsq_output.info)
    print("       sd_beta:", lsq_output.sd_beta)
    print("sqrt(diag(cov):", np.sqrt(np.diag(lsq_output.cov_beta)))

a_lsq, c_lsq = lsq_output.beta
print("        ODR(lsq)    curve_fit")
print("------------------------------")
print("a = %12.7f %12.7f" % (a_lsq, a_opt))
print("c = %12.7f %12.7f" % (c_lsq, c_opt))

odr = ODR(data, model, [3, -2])
odr.set_job(fit_type=0)
odr_output = odr.run()
print("Iteration 1:")
print("------------")
print("   stop reason:", odr_output.stopreason)
print("        params:", odr_output.beta)
print("          info:", odr_output.info)
print("       sd_beta:", odr_output.sd_beta)
print("sqrt(diag(cov):", np.sqrt(np.diag(odr_output.cov_beta)))

# if convergence is not reached, run again the algorithm
if odr_output.info != 1:
    print("\nRestart ODR till convergence is reached")
    i = 1
    while odr_output.info != 1 and i < 100:
        print("restart", i)
        odr_output = odr.restart()
        i += 1
    print("   stop reason:", odr_output.stopreason)
    print("        params:", odr_output.beta)
    print("          info:", odr_output.info)
    print("       sd_beta:", odr_output.sd_beta)
    print("sqrt(diag(cov):", np.sqrt(np.diag(odr_output.cov_beta)))

# Print the results and compare to least square
a_odr, c_odr = odr_output.beta
print("\n        ODR(lsq)    curve_fit     True ODR")
print("--------------------------------------------")
print("a = %12.7f %12.7f %12.7f" % (a_lsq, a_opt, a_odr))
print("c = %12.7f %12.7f %12.7f" % (c_lsq, c_opt, c_odr))

x = np.linspace(0, 20, 200)
ax = df.plot(
    x="x", y="y",
    kind="line", yerr="Dy", xerr="Dx",
    title="Some experimetal data", 
    linestyle="", marker=".",
    capthick=1, ecolor="gray", linewidth=1
)
ax.plot(x, f_model(x, a_lsq, c_lsq), linewidth=1, label="least square")
ax.plot(x, f_model(x, a_odr, c_odr), linewidth=1, label="ODR")
ax.legend(fontsize=14, frameon=True)

