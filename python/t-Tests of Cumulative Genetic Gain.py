import scipy.stats as stats

sample_stats = {
    '01_00': {
        'all': {
            'crispr':  {'s_mean':303138.417293, 's_std':8416.7110846, 's_n':10},
            'noedits': {'s_mean':342187.213993, 's_std':9728.71691188, 's_n':10},
            'perfect': {'s_mean':346087.672725, 's_std':9783.80876812, 's_n':10},
            'talen':   {'s_mean':292069.678028, 's_std':8078.1479302, 's_n':10},
            'zfn':     {'s_mean':289700.887676, 's_std':7977.01337275, 's_n':10},
            },
        'horned': {
            'crispr':  {'s_mean':302564.636384, 's_std':8403.58223586, 's_n':10},
            'noedits': {'s_mean':344956.580954, 's_std':9791.1546936, 's_n':10},
            'perfect': {'s_mean':345865.062233, 's_std':9786.05060032, 's_n':10},
            'talen':   {'s_mean':295344.085676, 's_std':8139.79102788, 's_n':10},
            'zfn':     {'s_mean':292997.457988, 's_std':7990.78092992, 's_n':10},
            },
    },
    '10_01': {
        'all': {
            'crispr':  {'s_mean':276025.745195, 's_std':7569.17149274, 's_n':10},
            'noedits': {'s_mean':345283.385014, 's_std':9768.11499647, 's_n':10},
            'perfect': {'s_mean':346050.437728, 's_std':9804.53554549, 's_n':10},
            'talen':   {'s_mean':251703.779485, 's_std':6859.47794086, 's_n':10},
            'zfn':     {'s_mean':235045.754069, 's_std':6309.36132342, 's_n':10},
            },
        'horned': {
            'crispr':  {'s_mean':274945.65872, 's_std':7559.8885364, 's_n':10},
            'noedits': {'s_mean':344325.284815, 's_std':9770.60946029, 's_n':10},
            'perfect': {'s_mean':344335.68879, 's_std':9804.53554549, 's_n':10},
            'talen':   {'s_mean':251252.190607, 's_std':9748.58963668, 's_n':10},
            'zfn':     {'s_mean':234885.991949, 's_std':6311.27201115, 's_n':10},
            },
    },
}

for k1 in sample_stats.keys():
    for k2 in sample_stats['01_00'].keys():
        for k3 in ['zfn', 'talen', 'crispr', 'perfect']:
            t2, p2 = stats.ttest_ind_from_stats(
                    sample_stats[k1][k2]['noedits']['s_mean'],
                    sample_stats[k1][k2]['noedits']['s_std'],
                    sample_stats[k1][k2]['noedits']['s_n'],
                    sample_stats[k1][k2][k3]['s_mean'],
                    sample_stats[k1][k2][k3]['s_std'],
                    sample_stats[k1][k2][k3]['s_n'],
                    equal_var=False
                )
            s_diff = sample_stats[k1][k2][k3]['s_mean'] -                      sample_stats[k1][k2]['noedits']['s_mean']
            print("%s %s %s: d = %g  t = %g  p = %g" % (k1.ljust(6),
                                                          k2.ljust(8),
                                                          k3.ljust(7),
                                                          s_diff,
                                                          t2,
                                                          p2)
                 )

for k1 in ['all', 'horned']:
    for k2 in sample_stats['01_00']['all'].keys():
        t2, p2 = stats.ttest_ind_from_stats(
            sample_stats['01_00'][k1][k2]['s_mean'],
            sample_stats['01_00'][k1][k2]['s_std'],
            sample_stats['01_00'][k1][k2]['s_n'],
            sample_stats['10_01'][k1][k2]['s_mean'],
            sample_stats['10_01'][k1][k2]['s_std'],
            sample_stats['10_01'][k1][k2]['s_n'],
            equal_var=False
        )
        s_diff = sample_stats['10_01'][k1][k2]['s_mean'] -                  sample_stats['01_00'][k1][k2]['s_mean']
        print("%s\t%s: d = %g  t = %g  p = %g" % (k1.ljust(6),
                                           k2.ljust(8),
                                           s_diff,
                                           t2,
                                           p2)
             )



