get_ipython().run_line_magic('load_ext', 'Cython')
# Set ourselves up for some plotting, too
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

get_ipython().run_cell_magic('cython', '-3 -lgsl -lgslcblas -lm', '\nimport msprime\nimport numpy as np\nimport struct\ncimport numpy as np\nfrom cython.view cimport array as cvarray\nfrom libc.stdlib cimport malloc, realloc, free\nfrom libc.stdint cimport int32_t, uint32_t\n\nfrom cython_gsl.gsl_rng cimport gsl_rng\nfrom cython_gsl.gsl_rng cimport gsl_rng_mt19937\nfrom cython_gsl.gsl_rng cimport gsl_rng_alloc\nfrom cython_gsl.gsl_rng cimport gsl_rng_free\nfrom cython_gsl.gsl_rng cimport gsl_rng_set\nfrom cython_gsl.gsl_rng cimport gsl_rng_uniform\nfrom cython_gsl.gsl_random cimport gsl_ran_flat\nfrom cython_gsl.gsl_random cimport gsl_ran_poisson\nfrom cython_gsl.gsl_vector cimport *\nfrom cython_gsl.gsl_sort cimport gsl_sort_vector\n\ncdef int32_t * malloc_int32_t(size_t n):\n    return <int32_t*>malloc(n*sizeof(int32_t))\n\ncdef int32_t * realloc_int32_t(void * x, size_t n):\n    return <int32_t*>realloc(x,n*sizeof(int32_t))\n\ncdef double * malloc_double(size_t n):\n    return <double*>malloc(n*sizeof(double))\n\ncdef double * realloc_double(void * x, size_t n):\n    return <double*>realloc(<double *>x,n*sizeof(double))\n\ncdef struct Mutations:\n    double * pos\n    int32_t * time\n    int32_t * node\n    size_t next_mutation, capacity\n    \ncdef int init_Mutations(Mutations * m):\n    m.next_mutation = 0\n    m.capacity = 10000\n    m.pos = malloc_double(m.capacity)\n    if m.pos == NULL:\n        return -1\n    m.time = malloc_int32_t(m.capacity)\n    if m.time == NULL:\n        return -1\n    m.node = malloc_int32_t(m.capacity)\n    if m.node == NULL:\n        return -1\n    return 0\n\ncdef int realloc_Mutations(Mutations * m):\n    m.capacity *= 2\n    m.pos = realloc_double(m.pos,\n                          m.capacity)\n    if m.pos == NULL:\n        return -1\n    m.time = realloc_int32_t(m.time,\n                            m.capacity)\n    if m.time == NULL:\n        return -1\n    m.node = realloc_int32_t(m.node,\n                            m.capacity)\n    if m.node == NULL:\n        return -1\n    return 0\n\ncdef void free_Mutations(Mutations * m):\n    free(m.pos)\n    free(m.time)\n    free(m.node)\n    m.next_mutation = 0\n    m.capacity = 10000\n    \ncdef int add_mutation(const double pos,\n                     const int32_t generation,\n                     const int32_t node,\n                     list metadata,\n                     Mutations * m):\n    cdef int rv = 0\n    if m.next_mutation+1 >= m.capacity:\n        rv = realloc_Mutations(m)\n        if rv != 0:\n            return rv\n    m.pos[m.next_mutation] = pos\n    m.time[m.next_mutation] = generation\n    m.node[m.next_mutation] = node\n    m.next_mutation+=1\n    metadata.append(struct.pack(\'id\',generation,pos))\n    return rv\n    \ncdef struct Nodes:\n    double * time\n    size_t next_node, capacity\n    \ncdef int init_Nodes(Nodes * n):\n    n.next_node = 0\n    n.capacity = 10000\n    n.time = malloc_double(n.capacity)\n    if n.time == NULL:\n        return -1\n    return 0\n\ncdef int realloc_Nodes(Nodes * n):\n    n.capacity *= 2\n    n.time = realloc_double(n.time,\n                            n.capacity)\n    if n.time == NULL:\n        return -1\n    return 0\n    \ncdef void free_Nodes(Nodes * n):\n    if n.time != NULL:\n        free(n.time)\n    n.next_node = 0\n    n.capacity = 10000\n\ncdef int add_node(const double t, Nodes *n):\n    cdef int rv = 0\n    if n.next_node >= n.capacity:\n        rv = realloc_Nodes(n)\n        if rv != 0:\n            return rv\n    n.time[n.next_node] = t\n    n.next_node+=1\n    return rv\n    \ncdef struct Edges:\n    double *left\n    double *right\n    int32_t *parent\n    int32_t *child\n    size_t next_edge, capacity\n    \ncdef int init_Edges(Edges * e):\n    e.next_edge = 0\n    e.capacity = 10000\n    e.left = malloc_double(e.capacity)\n    if e.left == NULL:\n        return -1\n    e.right = malloc_double(e.capacity)\n    if e.right == NULL:\n        return -1\n    e.parent = malloc_int32_t(e.capacity)\n    if e.parent == NULL:\n        return -1\n    e.child = malloc_int32_t(e.capacity)\n    if e.child == NULL:\n        return -1\n    return 0\n   \ncdef int realloc_Edges(Edges * e):\n    e.capacity *= 2\n    e.left = realloc_double(e.left,e.capacity)\n    if e.left == NULL:\n        return -1\n    e.right = realloc_double(e.right,e.capacity)\n    if e.right == NULL:\n        return -1\n    e.parent = realloc_int32_t(e.parent,e.capacity)\n    if e.parent == NULL:\n        return -1\n    e.child = realloc_int32_t(e.child,e.capacity)\n    if e.child == NULL:\n        return -1\n    return 0\n\ncdef void free_Edges(Edges * e):\n    free(e.left)\n    free(e.right)\n    free(e.parent)\n    free(e.child)\n    e.next_edge = 0\n    e.capacity = 10000\n    \ncdef int add_edge(const double left,const double right,\n             const int32_t parent,const int32_t child,\n             Edges * edges):\n    cdef int rv=0\n    if edges.next_edge+1 >= edges.capacity:\n        rv = realloc_Edges(edges)\n        if rv != 0:\n            return rv\n        \n    edges.left[edges.next_edge] = left\n    edges.right[edges.next_edge] = right\n    edges.parent[edges.next_edge] = parent\n    edges.child[edges.next_edge] = child\n    edges.next_edge += 1\n    return rv\n\ncdef struct Tables:\n    Nodes nodes\n    Edges edges\n    Mutations mutations\n    gsl_rng * rng\n    \ncdef int init_Tables(Tables * t, int seed):\n    cdef int rv = 0\n    rv = init_Nodes(&t.nodes)\n    if rv != 0:\n        return rv\n    rv = init_Edges(&t.edges)\n    if rv != 0:\n        return rv\n    rv = init_Mutations(&t.mutations)\n    if rv != 0:\n        return rv\n    t.rng = gsl_rng_alloc(gsl_rng_mt19937)\n    if t.rng == NULL:\n        return -1\n    gsl_rng_set(t.rng, seed)\n    return rv\n\ncdef void free_Tables(Tables * t):\n    free_Nodes(&t.nodes)\n    free_Edges(&t.edges)\n    free_Mutations(&t.mutations)\n    gsl_rng_free(t.rng)\n    \ncdef int infsites(const double mu,\n                  const int32_t generation,\n                  const int32_t next_offspring_index,\n                  Tables * tables,\n                  list metadata,\n                  dict lookup):\n    cdef unsigned nmut = gsl_ran_poisson(tables.rng, mu)\n    cdef unsigned i = 0\n    cdef double pos\n    cdef int rv = 0\n    for i in range(nmut):\n        pos = gsl_rng_uniform(tables.rng)\n        while pos in lookup:\n            pos = gsl_rng_uniform(tables.rng)\n        rv = add_mutation(pos,\n                         generation,\n                         next_offspring_index,\n                         metadata,\n                         &tables.mutations)\n        if rv != 0:\n            return rv\n        lookup[pos] = True\n    return rv\n\ncdef int value_present_vector(gsl_vector * v, double x,\n                              size_t start, size_t stop):\n    cdef size_t i\n    for i in range(start,stop):\n        if gsl_vector_get(v,i)==x:\n            return 1\n    return 0\n\ncdef int poisson_recombination(const double r,\n                               size_t pg1, size_t pg2,\n                               const int32_t next_offspring_id,\n                               Tables * tables):\n    cdef unsigned nbreaks = gsl_ran_poisson(tables.rng, r)\n    cdef gsl_vector * b = NULL\n    cdef unsigned i = 0#,drew_zero=0\n    cdef double x\n    cdef int rv = 0\n    cdef double left,right\n    cdef int32_t p\n    if nbreaks == 0:\n        # The parent passes the \n        # entire region onto the child\n        rv = add_edge(0.0,1.0,pg1,\n                      next_offspring_id,\n                      &tables.edges)\n        if rv != 0:\n            return rv\n    else:\n        b = gsl_vector_calloc(nbreaks+2) \n        while i < nbreaks:\n            x = gsl_rng_uniform(tables.rng)\n            while value_present_vector(b,x,0,i)==1:\n                x = gsl_rng_uniform(tables.rng)\n            gsl_vector_set(b,i,x)\n            i += 1\n        if gsl_vector_get(b,0) == 0.0:\n            pg1,pg2 = pg2,pg1\n            # We already have a zero\n            # in there, so we need\n            # to adjust size so that the \n            # 1.0 we insert below goes \n            # into the right place\n            b.size -= 1\n        else:\n            # shift all values by 1\n            # index and set element\n            # 0 to 0\n            # for i in range(b.size):\n            #     print(gsl_vector_get(b,i))\n            # print("-----")\n            for i in range(b.size-1,0,-1):\n                gsl_vector_set(b,i,\n                              gsl_vector_get(b,i-1))\n            gsl_vector_set(b,0,0.0)\n                \n        gsl_vector_set(b,b.size-1,1.0)\n        gsl_sort_vector(b)\n        # print("nbreaks=",nbreaks)\n        # for i in range(b.size):\n        #     print(gsl_vector_get(b,i))\n        # print("//")\n        # if drew_zero == 1:\n        #     pg1,pg2 = pg2,pg1\n        for i in range(b.size-1):\n            left = gsl_vector_get(b,i)\n            right = gsl_vector_get(b,i+1)\n            rv = add_edge(left,right,pg1,\n                          next_offspring_id,\n                          &tables.edges)\n            if rv != 0:\n                gsl_vector_free(b)\n                return rv\n            pg1,pg2 = pg2,pg1\n    gsl_vector_free(b)\n    return 0\n\ncdef int make_offspring(const double mu, const double r,\n                        const size_t generation,\n                        size_t pg1, size_t pg2,\n                        const int32_t next_offspring_index,\n                        list metadata,\n                        dict lookup,\n                        Tables * tables):\n    cdef int rv\n    rv = poisson_recombination(r,pg1,pg2,\n                               next_offspring_index,\n                               tables)\n    if rv != 0:\n        return -2\n                \n    rv = infsites(mu,generation+1,\n                  next_offspring_index,\n                  tables,metadata,lookup)\n    if rv != 0:\n        return -3\n            \n    rv = add_node(generation+1, &tables.nodes)\n    if rv != 0:\n        return -4\n   \n    return 0\n\ncdef void handle_error_code(int error, Tables * tables):\n    """\n    Only to be called after make_offspring\n    """\n    if error == 0:\n        return\n    print("Error occurred")\n    free_Tables(tables)\n    if error == -2:\n        raise RuntimeError("error during recombination")\n    elif error == -2:\n        raise RuntimeError("error during mutation")\n    elif error == -4:\n        raise RuntimeError("erorr adding nodes")\n    else:\n        raise ValueError("invalid error code")\n        \ncdef struct ParentalGametes:\n    int32_t p1g1\n    int32_t p1g2\n    int32_t p2g1\n    int32_t p2g2\n    \ncdef int pick_parents(const gsl_rng * r,\n                      const int32_t N,\n                      const int32_t first_parental_index,\n                      ParentalGametes * pg):\n    cdef int32_t p = <int32_t>gsl_ran_flat(r,0.0,<double>N)\n    pg.p1g1 = first_parental_index + 2*p\n    pg.p1g2 = pg.p1g1+1\n    # Mendel\n    if gsl_rng_uniform(r) < 0.5:\n        pg.p1g1,pg.p1g2 = pg.p1g2,pg.p1g1 \n    p = <int32_t>gsl_ran_flat(r,0.0,<double>N)\n    pg.p2g1 = first_parental_index + 2*p\n    pg.p2g2 = pg.p2g1+1\n    # Mendel\n    if gsl_rng_uniform(r) < 0.5:\n        pg.p2g1,pg.p2g2 = pg.p2g2,pg.p2g1 \n\ncdef int simplify(Tables * tables, \n            const double dt,\n            list metadata,\n            object nodes,\n            object edges,\n            object sites,\n            object mutations):\n    cdef int rv = 0,gap\n    \n    if tables.edges.next_edge == 0:\n        return rv\n    \n    cdef size_t i\n    cdef np.ndarray[double,ndim=1] dview,lview,rview\n    cdef np.ndarray[int32_t,ndim=1] pview,cview\n    # Reverse time for our new nodes\n    cdef gsl_vector_view vt\n    vt = gsl_vector_view_array(tables.nodes.time,<size_t>tables.nodes.next_node)\n    cdef double tmax,tmin\n    gsl_vector_minmax(&vt.vector,&tmin,&tmax)\n    for i in range(tables.nodes.next_node):\n        tables.nodes.time[i] = -1.0*(tables.nodes.time[i]-tmax)\n    gsl_vector_minmax(&vt.vector,&tmin,&tmax)\n    \n    nodes.set_columns(time=nodes.time+dt,\n                      flags=nodes.flags)\n    gap=nodes.time.min()-tmax\n    if gap != 1:\n        return -1\n    dview = np.asarray(<double[:tables.nodes.next_node]>tables.nodes.time)\n    nodes.append_columns(time=dview,\n                        flags=np.ones(tables.nodes.next_node,dtype=np.uint32))\n    \n    lview = np.asarray(<double[:tables.edges.next_edge]>tables.edges.left)\n    rview = np.asarray(<double[:tables.edges.next_edge]>tables.edges.right)\n    pview = np.asarray(<int32_t[:tables.edges.next_edge]>tables.edges.parent)\n    cview = np.asarray(<int32_t[:tables.edges.next_edge]>tables.edges.child)\n    edges.append_columns(left=lview,\n                        right=rview,\n                        parent=pview,\n                        child=cview)\n    \n    # We are trying to be as fast as possible here,\n    # so we\'ll use the more cumbersome \n    # append_columns interface instead of the \n    # much slower (but easier to understand)\n    # add_rows\n    cdef size_t nsites = len(sites)\n    if tables.mutations.next_mutation > 0:\n        encoded, offset = msprime.pack_bytes(metadata)\n        # for i in range(tables.mutations.next_mutation):\n        #     sites.add_row(position=tables.mutations.pos[i],\n        #                  ancestral_state=\'0\')\n        #     mutations.add_row(site=nsites+i,\n        #                      node=tables.mutations.node[i],\n        #                      derived_state=\'1\',\n        #                      metadata=metadata[i])\n        dview = np.asarray(<double[:tables.mutations.next_mutation]>tables.mutations.pos)\n        sites.append_columns(position=dview,\n                            ancestral_state=np.zeros(len(dview),dtype=np.int8)+ord(\'0\'),\n                            ancestral_state_offset=np.arange(len(dview)+1,dtype=np.uint32))\n        pview = np.asarray(<int32_t[:tables.mutations.next_mutation]>tables.mutations.node)\n        mutations.append_columns(site=np.arange(nsites,\n                                                nsites+tables.mutations.next_mutation,\n                                                dtype=np.int32),\n                                node=pview,\n                                derived_state=np.ones(len(dview),\n                                                      dtype=np.int8)+ord(\'0\'),\n                                derived_state_offset=np.arange(len(dview)+1,\n                                                              dtype=np.uint32),\n                                metadata_offset=offset, metadata=encoded\n                                )\n    \n    msprime.sort_tables(nodes=nodes,edges=edges,\n                       sites=sites,mutations=mutations)\n    samples = np.where(nodes.time == 0)[0]\n    msprime.simplify_tables(samples=samples.tolist(),\n                            nodes=nodes,\n                            edges=edges,\n                            sites=sites,\n                            mutations=mutations)\n    \n    # "clear" our temp containers\n    tables.nodes.next_node = 0\n    tables.mutations.next_mutation = 0\n    tables.edges.next_edge = 0\n                          \n    return rv\n\ndef evolve(int N, int ngens, double theta, double rho, int gc, int seed):\n    nodes = msprime.NodeTable()\n    edges = msprime.EdgeTable()\n    sites = msprime.SiteTable()\n    mutations = msprime.MutationTable()\n    \n    cdef double mu = theta/<double>(4*N)\n    cdef double r = rho/<double>(4*N)\n    cdef int rv\n    cdef size_t index=0,generation=0\n    cdef Tables tables\n    rv = init_Tables(&tables, seed)\n    if rv != 0:\n        free_Tables(&tables)\n        raise RuntimeError("could not initialize tables")\n    \n    nodes.set_columns(time=np.zeros(2*N),\n                     flags=np.ones(2*N,dtype=np.uint32))\n        \n    cdef int32_t next_offspring_index = len(nodes)\n    cdef int32_t first_parental_index = 0\n    cdef dict lookup = {}\n    cdef list metadata = []\n    cdef size_t last_gen_gc = 0\n    cdef ParentalGametes pgams\n    cdef double spos\n    for generation in range(<size_t>(ngens)):\n        if generation>0 and generation%gc == 0.0:\n            rv = simplify(&tables,\n                         generation-last_gen_gc,\n                         metadata,\n                         nodes,edges,sites,mutations)\n            if rv != 0:\n                free_Tables(&tables)\n                raise RuntimeError("simplification error")\n            lookup = {spos:True for spos in sites.position}\n            metadata.clear()\n            last_gen_gc=generation\n            next_offspring_index = len(nodes)\n            first_parental_index = 0\n        else:\n            first_parental_index = next_offspring_index - 2*N\n            \n        for index in range(N):\n            pick_parents(tables.rng,N,\n                         first_parental_index,\n                         &pgams)\n            rv = make_offspring(mu,r,generation,\n                                pgams.p1g1,pgams.p1g2,\n                                next_offspring_index,\n                                metadata,\n                                lookup,\n                                &tables)\n            handle_error_code(rv,&tables)\n            next_offspring_index+=1\n            rv = make_offspring(mu,r,generation,\n                                pgams.p2g1,pgams.p2g2,\n                                next_offspring_index,\n                                metadata,\n                                lookup,\n                                &tables)\n            assert(len(lookup) >=len(sites))\n            handle_error_code(rv,&tables)\n            next_offspring_index+=1\n            \n    if tables.nodes.next_node > 0:\n        rv=simplify(&tables,\n                    generation+1-last_gen_gc,\n                    metadata,\n                    nodes,edges,sites,mutations)\n        if rv == -1:\n            free_Tables(&tables)\n            raise RuntimeError("simplification error")\n    \n    free_Tables(&tables)\n    return msprime.load_tables(nodes=nodes,edges=edges,\n                               sites=sites,mutations=mutations)')

get_ipython().run_cell_magic('time', '', 'ts = evolve(100, 1000, 100.0, 100.0, 1, 42)')

for gc in range(10,1000,29):
    ts2 = evolve(100, 1000, 100.0, 100.0, gc, 42)
    assert(ts2.tables.nodes == ts.tables.nodes)
    assert(ts2.tables.edges == ts.tables.edges)
    assert(ts2.tables.sites == ts.tables.sites)
    assert(ts2.tables.mutations == ts.tables.mutations)

get_ipython().run_cell_magic('time', '', 'ts = evolve(1000,10000,100.0,1000.0,1000,42)')

get_ipython().run_cell_magic('prun', '-l 10 -s cumulative', 'ts = evolve(1000,10000,100.0,1000.0,1000,42)')

mdata = msprime.unpack_bytes(ts.tables.mutations.metadata,
                             ts.tables.mutations.metadata_offset)

for i in mdata:
    md = struct.unpack('id',i)

from IPython.display import SVG
import msprime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from libsequence.polytable import SimData
from libsequence.summstats import PolySIM
from libsequence.msprime import make_SimData
import concurrent.futures
import pandas as pd
from collections import namedtuple

SummStats=namedtuple('SummStats',['S','pi','D','hprime','rmin'])

# Simulate data with msprime
ts = msprime.simulate(10,mutation_rate=1,random_seed=666)

# Get it into the format expected by pylibseq
d = make_SimData(ts)

# This should look familiar! :)
print(d)

# Create object to calculate summary stats
x = PolySIM(d)
# Calculate a few:
print(x.thetapi(),x.tajimasd(),x.hprime(),x.rm())

get_ipython().run_cell_magic('time', '', 'msprime_raw_data=[]\nfor i in msprime.simulate(10,mutation_rate=100.0/4.0,\n                          recombination_rate=0.0/4., #100.0/4.0,\n                          num_replicates=1000,\n                          random_seed=42):\n    d = make_SimData(i)\n    ps = PolySIM(d)\n    # A little check that the two pieces of code agree\n    assert(ps.numpoly() == i.num_mutations)\n    msprime_raw_data.append(SummStats(ps.numpoly(),\n                                      ps.thetapi(),ps.tajimasd(),\n                                      ps.hprime(),ps.rm()))')

def run_forward_sim(nreps,seed,repid):
    """
    Run our forward sim, calculate
    a bunch of stats, and return 
    the list.
    """
    # Not the best seeding scheme, 
    # but good enough for now...
    np.random.seed(seed)
    msp_rng = msprime.RandomGenerator(int(seed))
    seeds = np.random.randint(0,1000000,nreps) * repid
    sims = []
    for i in range(nreps):
        ts = evolve(500,40000,0.0,0.0,500,seeds[i])
        samples = np.random.choice(1000,10,replace=False)
        assert(all(ts.tables.nodes.time[samples]==0.0))
        ts2 = ts.simplify(samples=samples.tolist())
        n=msprime.NodeTable()
        e=msprime.EdgeTable()
        s=msprime.SiteTable()
        m=msprime.MutationTable()
        ts2.dump_tables(nodes=n,edges=e)
        mutgen = msprime.MutationGenerator(
            msp_rng, 100.0/(float(4*500)))
        mutgen.generate(n,e,s,m)
        ts2=msprime.load_tables(nodes=n,edges=e,sites=s,mutations=m)
        # print(samples)
        # print(n.time[samples])
        # print(s)
        # print(m)
        # Simplify from entire pop down
        # to random sample of n << 2N
        # slist = samples.tolist()
        # slist.append(8)
        # ts2=ts.simplify(slist)
        # print(ts2.num_mutations)
        # print(len(ts2.tables.nodes),
        #      len(ts2.tables.edges),
        #      len(ts2.tables.sites),
        #      len(ts2.tables.mutations))
        d=make_SimData(ts2)
        ps=PolySIM(d)
        sims.append(SummStats(ps.numpoly(),
                              ps.thetapi(),
                              ps.tajimasd(),
                              ps.hprime(),
                              ps.rm()))
    return sims

get_ipython().run_cell_magic('time', '', 'x=run_forward_sim(1,42,3511)\nprint(x)')

get_ipython().run_cell_magic('time', '', 'fwd_sim_data=[]\nnp.random.seed(16463623)\nwith concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:\n    futures = {executor.submit(run_forward_sim,50,np.random.randint(0,2000000,1)[0],i): i for i in range(4)}\n    for fut in concurrent.futures.as_completed(futures):\n        fn = fut.result()\n        fwd_sim_data.extend(fn)')

msprime_df = pd.DataFrame(msprime_raw_data)
msprime_df['engine'] = ['msprime']*len(msprime_df.index)
fwd_df = pd.DataFrame(fwd_sim_data)
fwd_df['engine']=['forward']*len(fwd_df)
summstats_df = pd.concat([msprime_df,fwd_df])

sns.set(style="darkgrid")
g = sns.FacetGrid(summstats_df,col="engine",margin_titles=True)
bins = np.linspace(summstats_df.pi.min(),summstats_df.pi.max(),20)
g.map(plt.hist,'pi',bins=bins,color="steelblue",lw=0,density=True);

g = sns.FacetGrid(summstats_df,col="engine",margin_titles=True)
bins = np.linspace(summstats_df.S.min(),summstats_df.S.max(),20)
g.map(plt.hist,'S',bins=bins,color="steelblue",lw=0,density=True);

g = sns.FacetGrid(summstats_df,col="engine",margin_titles=True)
bins = np.linspace(summstats_df.D.min(),summstats_df.D.max(),20)
g.map(plt.hist,'D',bins=bins,color="steelblue",lw=0,density=True);

g = sns.FacetGrid(summstats_df,col="engine",margin_titles=True)
bins = np.linspace(summstats_df.rmin.min(),summstats_df.rmin.max(),20)
g.map(plt.hist,'rmin',bins=bins,color="steelblue",lw=0,density=True);

len(fwd_df.index)

from scipy.stats import ks_2samp

print(summstats_df.groupby(['engine']).agg(['mean','std']))

ks_2samp(fwd_df.pi,msprime_df.pi)

ks_2samp(fwd_df.S,msprime_df.S)

ks_2samp(fwd_df.D,msprime_df.D)

ks_2samp(fwd_df.rmin,msprime_df.rmin)



