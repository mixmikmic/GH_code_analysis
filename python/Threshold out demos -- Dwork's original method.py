get_ipython().magic('matplotlib nbagg')
import supplementalDemo as original_demo

reps = 100
n, d = 5000, 500
krange = [0, 10, 20, 30, 45, 70, 100, 150, 200]

original_demo.runandplotsummary(n, d, krange, reps, original_demo.createnosignaldata,  'nosignal')

# Experiment 2:
# Some variables are correlated

original_demo.runandplotsummary(n, d, krange, reps, original_demo.createhighsignaldata,  'highsignal')



