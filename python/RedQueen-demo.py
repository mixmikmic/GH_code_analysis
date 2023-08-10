get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

np.set_printoptions(suppress=True, precision=3)

sns.set(style='ticks', palette='Set2')
sns.despine()

import redqueen.utils as U
import redqueen.opt_runs as OR
import redqueen.opt_model as OM
import decorated_options as Deco

sim_opts_1_follower = OM.SimOpts(
    src_id=0,
    end_time=100,
    q_vec=np.array([1]),
    s=1.0,
    other_sources=[('Hawkes', {'src_id': 1, 'seed': 1, 'l_0': 1.0, 'alpha': 1.0, 'beta': 5.0})],
    sink_ids=[1000],
    edge_list=[(0, 1000), (1, 1000)]
)

get_ipython().run_cell_magic('time', '', "seed = 1\nopt_mgr = sim_opts_1_follower.create_manager_with_opt(seed)\nopt_mgr.run_dynamic()\nopt_df = opt_mgr.state.get_dataframe()\nnum_opt_tweets = U.num_tweets_of(opt_df, broadcaster_id=0)\nperf_opt = {\n    'type': 'Opt',\n    'seed': seed,\n    'capacity': num_opt_tweets,\n    's': sim_opts_1_follower.s\n}\nOR.add_perf(perf_opt, opt_df, sim_opts_1_follower)")

get_ipython().run_cell_magic('time', '', "seed = 9\npoisson_mgr = sim_opts_1_follower.create_manager_with_poisson(seed, capacity=num_opt_tweets)\npoisson_mgr.run_dynamic()\npoisson_df = poisson_mgr.state.get_dataframe()\nnum_poisson_tweets = U.num_tweets_of(opt_df, broadcaster_id=0)\nperf_poisson = {\n    'type': 'Poisson',\n    'seed': seed,\n    'capacity': num_poisson_tweets,\n    's': sim_opts_1_follower.s\n}\nOR.add_perf(perf_poisson, poisson_df, sim_opts_1_follower)")

print('num_opt_tweets = {}, num_poisson_tweets = {}'
      .format(U.num_tweets_of(opt_df, 0), U.num_tweets_of(poisson_df, 0)))

print('avg_rank_opt = {}, avg_rank_poisson = {}'
      .format(U.average_rank(opt_df, sim_opts=sim_opts_1_follower), 
              U.average_rank(poisson_df, sim_opts=sim_opts_1_follower)))

perf_opt

print('top_1_opt = {}, top_1_poisson = {}'
      .format(perf_opt['top_1'], perf_poisson['top_1']))

perf_opt

tmp = U.rank_of_src_in_df(opt_df, 0).mean(1)
list(zip(tmp.index, tmp.values))

@Deco.optioned('opts')
def perf_to_json(dfs, names, src_id, sink_ids, end_time):
    """Produce a dictionary which captures performance for the demo app."""        
    
    # Assumes that the walls across the data frames are the same.
    eg_df = dfs[0]
    
    walls = {}
    for sink_id in sink_ids:
        walls[sink_id] = eg_df[(eg_df.src_id != src_id) & (eg_df.sink_id == sink_id)].t.tolist()
    
    broadcasts = {}
    for df, name in zip(dfs, names):
        r_t = U.rank_of_src_in_df(df, src_id)
        avg_rank = r_t.mean(1)
        time_at_top = np.where(r_t < 1.0, 1.0, 0.0).mean(1)
        dt = np.diff(np.concatenate([r_t.index.values, [end_time]]))
        broadcasts[name] = {
            'post_times': df[df.src_id == src_id].t.unique().tolist(),
            'performance': {
                'avg_rank': list(zip(avg_rank.index, avg_rank.values)),
                'time_at_top': list(zip(avg_rank.index, np.cumsum(time_at_top * dt)))
            }
        }
        
    return {
        'walls': walls,
        'broadcasts': broadcasts
    }

example_1 = perf_to_json([opt_df, poisson_df], ['redqueen', 'poisson'], 
                         opts=Deco.Options(**sim_opts_1_follower.get_dict()))

U.time_in_top_k(opt_df, src_id=1, K=1, sim_opts=sim_opts_1_follower)

example_1['broadcasts']

import json
with open('data/example2.json', 'w') as f:
    json.dump(example_1, f, indent=2)



















