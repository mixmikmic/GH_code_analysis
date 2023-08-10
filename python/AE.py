# preamble
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
get_ipython().magic('run lib/preamble.py')

complete(clgen.version() == "0.1.7", "Initial setup complete")

import clgen.clutil
clgen.clutil.platform_info()

import random
uid = random.randint(0, 100000)
fs.rm("../data/usr/{uid}".format(uid=uid))
fs.mkdir("../data/usr/{uid}/clgen".format(uid=uid))
fs.mkdir("../data/usr/{uid}/benchmarks".format(uid=uid))
print("\nUnique test ID:", uid)

complete(can_reproduce_experiments(), "Artifact is running on suitable hardware")

print("The model used in the paper (pre-trained):")
model = clgen.model.from_tar("../data/clgen-github-model-2016-nov-2048x3.tar.bz2")
print(model)
complete(model.hash == "f2fb3ad753896d54fe284c138eaa703db3518bbb",
         "Load pre-trained neural network")

# sample model
import clgen.sampler
import clgen.dbutil
import clgen.explore

argspec = ['__global float*', '__global float*', '__global float*', 'const int']
sampler = clgen.sampler.from_json({
        "kernels": { 
            "args": argspec,
            "max_length": 1000
        },
        "sampler": {
            "batch_size": 25,
            "max_kernels": 10
        }
    })

print("Sample from the model used in the paper:\n")
print("Seed text:", clgen.sampler.serialize_argspec(argspec), "\n")
sampler.cache(model).empty()
sampler.sample(model)

db = sampler.cache(model)["kernels.db"]
num_good_kernels = clgen.dbutil.num_good_kernels(db)
clgen.explore.explore(db)
complete(num_good_kernels >= 5, "Generated {} OpenCL kernels".format(num_good_kernels))

print("Generated kernels\n")
try:
    db = clgen.dbutil.connect(sampler.cache(model)["kernels.db"])
    c = db.cursor()

    c.execute("""SELECT Contents FROM PreprocessedFiles WHERE status=0""")
    for i, row in enumerate(c.fetchall()):
        kernel = row[0]
        print("\nKernel ", i+1, ":\n", sep="")
        print(kernel)

    c.close(); db.close()
    complete(msg="Display generated OpenCL kernels")
except:
    complete(False, "Failed to display generated OpenCL kernels")

print("running ...  (this will take a few minutes)")
try:
    get_ipython().system('rm -f ../data/benchmarks/*.csv ../data/benchmarks/timestamp.csv')
    get_ipython().system('cd benchmarks && ./mkdata')
    data = pd.read_csv("../data/benchmarks/training.csv")
    benchmarks_timestamp = readfile("../data/benchmarks/timestamp.txt")
    move("../data/benchmarks/training.csv", "../data/usr/{uid}/benchmarks/".format(uid=uid))
    move("../data/benchmarks/timestamp.txt", "../data/usr/{uid}/benchmarks/".format(uid=uid))
    complete(len(data) == 17, "Produced new performance results for benchmarks")
except:
    complete(False, "Did not produce new performance results for benchmarks")

try:
    if benchmarks_timestamp != readfile("../data/usr/{uid}/benchmarks/timestamp.txt".format(uid=uid)):
        print("warning: data timestamp has changed, please re-run experiments", file=sys.stderr)
    data = pd.read_csv("../data/usr/{uid}/benchmarks/training.csv".format(uid=uid))
    ax = sns.barplot(x="benchmark", y="speedup", data=data)
    plt.title("Runtimes generated " + benchmarks_timestamp)
    plt.ylabel("Max speedup")
    plt.xlabel("AMD SDK Benchmark kernels")
    plt.axhline(y=1, color="k", lw=1)  # speedup line
    plt.setp(ax.get_xticklabels(), rotation=90)  # rotate x ticks
    ax.set_xticklabels([shortbenchmark(x.get_text()) for x in ax.get_xticklabels()])
    viz.finalise(figsize=(9,4))
    complete(len(set(data["benchmark"])) == 17, "New performance numbers from 17 AMD kernels")
except:
    complete(False, "Failed to analyze benchmark results")

print("running ...  (this will take a few minutes)")
try:
    get_ipython().system('rm -f ../data/clgen-10/*.csv ../data/clgen-10/timestamp.txt')
    get_ipython().system('cd bin && ./mkdata')
    data = pd.read_csv("../data/clgen-10/training.csv")
    clgen_timestamp = readfile("../data/clgen-10/timestamp.txt")
    move("../data/clgen-10/training.csv", "../data/usr/{uid}/clgen/".format(uid=uid))
    move("../data/clgen-10/timestamp.txt", "../data/usr/{uid}/clgen/".format(uid=uid))
    complete(len(set(data["benchmark"])) == 17, "Produced new performance results for CLgen benchmarks")
except:
    complete(False, "Did not produce new performance results for CLgen benchmarks")

try:
    if clgen_timestamp != readfile("../data/usr/{uid}/clgen/timestamp.txt".format(uid=uid)):
        print("warning: data timestamp has changed, please re-run experiments", file=sys.stderr)

    data = pd.read_csv("../data/usr/{uid}/clgen/training.csv".format(uid=uid))   
    ax = sns.barplot(x="benchmark", y="speedup", ci=95, data=data)
    plt.title("Runtimes generated " + clgen_timestamp)
    plt.ylabel("Max speedups (95% CI across datasets)")
    plt.xlabel("CLgen kernels")
    plt.axhline(y=1, color="k", lw=1)  # speedup line
    ax.set_xticklabels(range(1, len(data) + 1))
    viz.finalise(figsize=(9,4))
    complete(len(set(data["benchmark"])) == 17, "New performance numbers from 17 CLgen kernels")
except:
    complete(False, "Failed to analyze CLgen benchmark results")

try:
    header("Results from the paper on AMD")
    plot_speedups_with_clgen("../data/amd-benchmarks.csv", "../data/amd-clgen.csv", suite="npb")

    header("Results from the paper on NVIDIA")
    plot_speedups_with_clgen("../data/nvidia-benchmarks.csv", "../data/nvidia-clgen.csv", suite="npb")

    header("Results using runtimes generated: Benchmarks",
           readfile("../data/usr/{uid}/benchmarks/timestamp.txt".format(uid=uid)), "- CLgen",
           readfile("../data/usr/{uid}/clgen/timestamp.txt".format(uid=uid)))
    a, b = plot_speedups_with_clgen("../data/usr/{uid}/benchmarks/training.csv".format(uid=uid),
                                    "../data/usr/{uid}/clgen/training.csv".format(uid=uid), suite="amd")
    complete(b > a, "Predictive mode performance improves with CLgen kernels by {:.0f}%".format((b / a) * 100 - 100))
except:
    complete(False, "Failed to generate data for predictive model")

try:
    header("Results from the paper")
    plot_speedups_extended_model_2platform(("../data/amd-benchmarks.csv", "../data/amd-clgen.csv"),
                                           ("../data/nvidia-benchmarks.csv", "../data/nvidia-clgen.csv"))

    header("Results using new data")
    speedup = plot_speedups_extended_model("../data/usr/{uid}/benchmarks/training.csv".format(uid=uid),
                                           "../data/usr/{uid}/clgen/training.csv".format(uid=uid))
    complete(speedup >= 1.0, "Extended predictie model improves performance by {:.0f}%".format(speedup * 100 - 100))
except:
    complete(False, "Failed to generate data for extended predictive model")

