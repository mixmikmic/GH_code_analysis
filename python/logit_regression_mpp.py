get_ipython().magic("run '../database_connectivity_setup.ipynb'")

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sns

sql = """create schema mcmc;"""
psql.execute(sql,conn)
conn.commit()

sql = """drop function if exists mcmc.sample_from_multivariate_normal(float8[], float8[], int);"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create or replace function mcmc.sample_from_multivariate_normal(
        mu float8[],
        cov float8[],
        ncols int
    )
    returns float8[]
    AS
    $$
        import numpy as np;
        from numpy.random import multivariate_normal
        mean = np.array(mu)
        cov_mat = np.matrix(cov).reshape(len(cov)/ncols, ncols)
        return multivariate_normal(mean=mean,cov=cov_mat,size =1).flatten().tolist()
    $$language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """drop function if exists mcmc.compute_matrix_inverse(float8[], int);"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create or replace function mcmc.compute_matrix_inverse(
        matrix float8[],
        ncols int
    )
    returns SETOF float8[]
    AS
    $$
        import numpy as np;
        mat = np.matrix(matrix).reshape(len(matrix)/ncols, ncols)
        plpy.info(mat.shape)
        return np.linalg.inv(mat).tolist()
    $$language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """DROP AGGREGATE IF EXISTS mcmc.array_agg_array(anyarray) CASCADE;"""
psql.execute(sql,conn)
conn.commit()

sql = """
    CREATE ORDERED AGGREGATE mcmc.array_agg_array(anyarray)
    (
        SFUNC = array_cat,
        STYPE = anyarray
    );
"""
psql.execute(sql,conn)
conn.commit()

sql = """drop function if exists mcmc.lin_regr_ols_estimate(varchar, varchar, varchar, varchar, varchar, varchar);"""
psql.execute(sql,conn)
conn.commit()

# Function arguments:
# x: name of the table that stores the data matrix
# x_rowid: row identifier column in the data matrix
# x_rowvec: row vector for a given row in the data matrix
# y: name of the table that stores the response vector
# y_rowid: row identifier column for each element in the response vector table
# y_rowvec: 1-D array storing each element in the response vector table

sql = """
    create or replace function mcmc.lin_regr_ols_estimate(
        x varchar,
        x_rowid varchar,
        x_rowvec varchar,
        y varchar,
        y_rowid varchar,
        y_rowvec varchar
    )
    returns float8[]
    AS
    $$
        plpy.execute('drop table if exists mcmc.x_t_x;')
        sqlstr = \"""
            select
                madlib.matrix_mult(
                    '{x_matrix}',
                    'row={xid}, val={xval}, trans=true',
                    '{x_matrix}',
                    'row={xid}, val={xval}',
                    'mcmc.x_t_x',
                    'row=row_id, val=row_vec'
                )
        \""".format(x_matrix=x, xid=x_rowid, xval=x_rowvec)
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists mcmc.x_t_y;')
        sqlstr = \"""
            select
                madlib.matrix_mult(
                    '{x_matrix}',
                    'row={xid}, val={xval}, trans=true',
                    '{y_vector}',
                    'row={yid}, val={yval}',
                    'mcmc.x_t_y',
                    'row=row_id, val=element'
                )
        \""".format(
                x_matrix=x, 
                xid=x_rowid, 
                xval=x_rowvec, 
                y_vector=y, 
                yid=y_rowid, 
                yval=y_rowvec
            )
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists x_t_x_inv;')
        sqlstr = \"""
            create temp table x_t_x_inv as (
                select
                    row_number() over() as row_id,
                    inv as row_vec
                from (
                    select
                        mcmc.compute_matrix_inverse (
                            m,
                            ncols
                        ) as inv 
                    from (
                        select
                            mcmc.array_agg_array(row_vec order by row_id) as m,
                            max(array_upper(row_vec, 1)) as ncols
                        from            
                            mcmc.x_t_x
                    )q1
                ) q
            ) distributed by (row_id);
        \"""
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists mcmc.lin_regr_ols_estimate;')
        sqlstr = \"""
            select
                madlib.matrix_mult(
                    'x_t_x_inv',
                    'row=row_id, val=row_vec',
                    'mcmc.x_t_y',
                    'row=row_id, val=element',
                     'mcmc.lin_regr_ols_estimate',
                    'row=row_id, val=element'
                )
        \"""
        plpy.execute(sqlstr)
        
        sqlstr = \"""
            select 
                array_agg(ele order by row_id) as ols 
            from (
                select
                    row_id,
                    unnest(element) as ele
                from
                    mcmc.lin_regr_ols_estimate
            )q
        \"""
        rv = plpy.execute(sqlstr)
        return rv[0]['ols']
    $$ language plpythonu;
"""
psql.execute(sql,conn)
conn.commit();

sql = """drop function if exists mcmc.create_identity_matrix(int);"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create or replace function mcmc.create_identity_matrix(
        n int
    )
    returns SETOF float8[]
    AS
    $$
        import numpy as np;
        return np.identity(n).tolist()
    $$language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """drop function if exists mcmc.multivariate_log_density(float8[], float8[], float8[], int);"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create or replace function mcmc.multivariate_log_density(
        vec float8[],
        mu float8[],
        cov float8[],
        ncols int
    )
    returns float8
    AS
    $$
        import numpy as np;
        from scipy.stats import multivariate_normal as scipymvn
        cov_mat = np.matrix(cov).reshape(len(cov)/ncols, ncols)
        return scipymvn.logpdf(x=vec,mean=mu,cov=cov_mat)
    $$language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """
    drop function if exists mcmc.MHLogit(
        varchar, 
        varchar,
        varchar,
        varchar,
        varchar,
        varchar,
        float8[], 
        varchar,
        varchar,
        varchar,
        int, 
        int
    );
"""
psql.execute(sql,conn)
conn.commit();

# Metropolis Hastings Logistic Regression Driver Function
# Arguments:
# x - name of the table storing the data matrix
# x_rowid - the row identifier column in the x data matrix
# x_rowvec - the vector per row in the x data matrix stored as an array
# y - name of the table storing the response column vector (binary column, should be 1 and 0)
# y_rowid : the row identifier for each element of the y vector. y_rowid should match x_rowid
# y_element: a 1-Dimensional array containing the y vector values
# prior_beta_mu : the mean of the normal beta prior distribution supplied as an array
# prior_beta_cov : name of the table storing the covariance matrix of the normal beta prior distribution
# prior_beta_cov_rowid : the row identifier column in the prior_beta_cov table
# prior_beta_cov_rowvec : the vector per row in the prior_beta_cov matrix stored as an array 
# num : number of Gibbs iterations
# num_burnin_iter: number of burnin iterations

sql = """
    create or replace function mcmc.MHLogit(
        x varchar,
        x_rowid varchar,
        x_rowvec varchar,
        y varchar,
        y_rowid varchar,
        y_element varchar,
        prior_beta_mu float8[],
        prior_beta_cov varchar,
        prior_beta_cov_rowid varchar,
        prior_beta_cov_rowvec varchar,
        num_iter int,
        num_burnin_iter int
    )
    returns text
    AS
    $$
        import numpy as np

        plpy.execute('drop table if exists prior_beta_cov_inv;')
        sqlstr = \"""
            create temp table prior_beta_cov_inv as (
                select
                    row_number() over() as row_id,
                    inv as row_vec
                from (
                    select
                        mcmc.compute_matrix_inverse (
                            m,
                            ncols
                        ) as inv 
                    from (
                        select
                            mcmc.array_agg_array({val} order by {rowid}) as m,
                            max(array_upper({val}, 1)) as ncols
                        from            
                            {covtable}
                    )q1
                ) q
            ) distributed by (row_id);
        \""".format(
                val=prior_beta_cov_rowvec, 
                rowid=prior_beta_cov_rowid,
                covtable=prior_beta_cov
            )
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists mcmc.x_t_x;')
        sqlstr = \"""
            select
                madlib.matrix_mult(
                    '{x_matrix}',
                    'row={xid}, val={xval}, trans=true',
                    '{x_matrix}',
                    'row={xid}, val={xval}',
                    'mcmc.x_t_x',
                    'row=row_id, val=row_vec'
                )
        \""".format(x_matrix=x, xid=x_rowid, xval=x_rowvec)
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists mcmc.add_inv_cov_mat_x_t_x;')
        sqlstr = \"""
            select
                madlib.matrix_add(
                    'prior_beta_cov_inv',
                    'row=row_id, val=row_vec',
                    'mcmc.x_t_x',
                    'row=row_id, val=row_vec',
                    'mcmc.add_inv_cov_mat_x_t_x',
                    'row=row_id, val=row_vec'
                )
        \"""
        plpy.execute(sqlstr)
        
        plpy.execute('drop table if exists b_tilda;')
        sqlstr = \"""
            create temp table b_tilda as (
                select
                    row_number() over() as row_id,
                    inv as row_vec
                from (
                    select
                        mcmc.compute_matrix_inverse (
                            m,
                            ncols
                        ) as inv 
                    from (
                        select
                            mcmc.array_agg_array(row_vec order by row_id) as m,
                            max(array_upper(row_vec, 1)) as ncols
                        from            
                            mcmc.add_inv_cov_mat_x_t_x
                    )q1
                ) q
            ) distributed by (row_id);
        \"""
        plpy.execute(sqlstr)
        
        sqlstr = \"""
            select max(array_upper({x_rowvec},1)) as num_features from {x_matrix};
        \""".format(x_rowvec=x_rowvec, x_matrix=x)
        rv = plpy.execute(sqlstr)
        num_features = rv[0]['num_features']
        
        plpy.execute('drop table if exists beta_vec;')
        plpy.execute(\"""
            create temp table beta_vec (row_id int, element float8[]) distributed by (row_id);
            \""")

        sqlstr = \"""
            insert into beta_vec(
                select 
                    row_id,
                    array[beta]
                from (
                    select
                        generate_series(1,{num_features}) as row_id,
                        unnest(ols) as beta
                    from (
                        select 
                            mcmc.lin_regr_ols_estimate(
                                '{x_matrix}', 
                                '{x_rowid}', 
                                '{x_rowvec}', 
                                '{y_vector}', 
                                '{y_rowid}', 
                                '{y_element}'
                            ) as ols
                    )q1
                )q2
            );
        \""".format(
                num_features=num_features,
                x_matrix=x,
                x_rowid=x_rowid,
                x_rowvec=x_rowvec,
                y_vector=y,
                y_rowid=y_rowid,
                y_element=y_element
        )
        plpy.execute(sqlstr)
        
        rv = plpy.execute(\"""
            select 
                array_agg(beta order by row_id) as beta
            from (
                select 
                    row_id,
                    unnest(element) as beta
                from beta_vec
            ) q;
        \""")
        beta = rv[0]['beta']

        plpy.info(beta)
        
        plpy.execute('drop table if exists mcmc.x_beta;')
        sqlstr = \"""
            select
                madlib.matrix_mult(
                    '{x_matrix}',
                    'row={xid}, val={xval}',
                    'beta_vec',
                    'row=row_id,val=element',
                    'mcmc.x_beta',
                    'row=row_id,val=element'
                );
        \""".format(
                x_matrix=x, 
                xid=x_rowid, 
                xval=x_rowvec
            )
        plpy.execute(sqlstr)
        
        sqlstr = \"""
            select 
                sum(ll) as loglikelihood 
            from (
                select
                    ((y*log(exp(1.0)::numeric,p::numeric)) + 
                        ((1.0-y)*log(exp(1.0)::numeric,(1.0-p)::numeric))) as ll
                from (
                    select
                        unnest(t1.{y_element}) as y,
                        1.0/(1.0 + exp(-1.0*unnest(t2.element))) as p
                    from
                        {y_vector} t1,
                        mcmc.x_beta t2
                    where
                        t1.{y_rowid} = t2.row_id
                ) q1
            )q2;
        \""".format(
            y_element = y_element,
            y_vector = y,
            y_rowid = y_rowid
        )

        rv = plpy.execute(sqlstr)
        loglikelihood = rv[0]['loglikelihood']

        sqlstr = \"""
            select
                mcmc.multivariate_log_density (
                    array{beta},
                    array{prior_beta_mu},
                    q1.cov,
                    q1.ncols
                ) as logprior 
            from (
                select
                    mcmc.array_agg_array({row_vec} order by {row_id}) as cov,
                    max(array_upper({row_vec}, 1)) as ncols
                from            
                    {prior_beta_cov}
            )q1;
            \""".format(
                    prior_beta_mu=prior_beta_mu,
                    prior_beta_cov=prior_beta_cov,
                    row_vec=prior_beta_cov_rowvec,
                    row_id=prior_beta_cov_rowid,
                    beta=beta
                )
        
        rv = plpy.execute(sqlstr)
        logprior = rv[0]['logprior']
        logposterior_curr = loglikelihood + logprior
        
        plpy.info(loglikelihood)
        plpy.info(logprior)
        plpy.info(logposterior_curr)

        num_accepts = 0
        coeffs = np.zeros(shape=(num_iter-num_burnin_iter,num_features))
        coefindx = 0
        plpy.execute('drop table if exists mcmc.mhtrace;')
        plpy.execute(\"""
            create table mcmc.mhtrace(iter_num int, beta float8[]) 
            distributed by (iter_num);
        \""")
        
        plpy.info('Entering Metropolis Hastings Iterations')
        
        for ite in range(0,num_iter):
            plpy.info('Iteration number: {ite}'.format(ite=ite))
        
            sqlstr = \"""
                select
                    mcmc.sample_from_multivariate_normal (
                        array{beta},
                        q1.cov,
                        q1.ncols
                    ) as sample 
                from (
                    select
                        mcmc.array_agg_array(row_vec order by row_id) as cov,
                        max(array_upper(row_vec, 1)) as ncols
                    from            
                        b_tilda
                )q1;
            \""".format(beta=beta)
            rv = plpy.execute(sqlstr)
            beta_can = rv[0]['sample'] # This will be in memory. Should be ok
            # as this has dimensions = number features.
            
            
            plpy.execute('drop table if exists beta_can_vec;')
            plpy.execute(\"""
                create temp table beta_can_vec (row_id int, element float8[]) 
                distributed by (row_id);
            \""")

            sqlstr = \"""
                insert into beta_can_vec(
                    select 
                        row_id,
                        array[beta]
                    from (
                        select
                            generate_series(1,{num_features}) as row_id,
                            unnest(beta_arr) as beta
                        from (
                            select array{beta_can} as beta_arr
                        )q1
                    )q2
                );
            \""".format(
                    num_features=num_features,
                    beta_can = beta_can
            )
            plpy.execute(sqlstr)
            
            plpy.execute('drop table if exists mcmc.x_beta;')
            sqlstr = \"""
                select
                    madlib.matrix_mult(
                        '{x_matrix}',
                        'row={xid}, val={xval}',
                        'beta_can_vec',
                        'row=row_id,val=element',
                        'mcmc.x_beta',
                        'row=row_id,val=element'
                    );
            \""".format(
                    x_matrix=x, 
                    xid=x_rowid, 
                    xval=x_rowvec
                )
            plpy.execute(sqlstr)

            sqlstr = \"""
                select 
                    sum(ll) as loglikelihood 
                from (
                    select
                        ((y*log(exp(1.0)::numeric,p::numeric)) + 
                            ((1.0-y)*log(exp(1.0)::numeric,(1.0-p)::numeric))) as ll
                    from (
                        select
                            unnest(t1.{y_element}) as y,
                            1.0/(1.0 + exp(-1.0*unnest(t2.element))) as p
                        from
                            {y_vector} t1,
                            mcmc.x_beta t2
                        where
                            t1.{y_rowid} = t2.row_id
                    ) q1
                )q2;
            \""".format(
                y_element = y_element,
                y_vector = y,
                y_rowid = y_rowid
            )

            rv = plpy.execute(sqlstr)
            loglikelihood = rv[0]['loglikelihood']

            sqlstr = \"""
                select
                    mcmc.multivariate_log_density (
                        array{beta_can},
                        array{prior_beta_mu},
                        q1.cov,
                        q1.ncols
                    ) as logprior 
                from (
                    select
                        mcmc.array_agg_array({row_vec} order by {row_id}) as cov,
                        max(array_upper({row_vec}, 1)) as ncols
                    from            
                        {prior_beta_cov}
                )q1;
                \""".format(
                        prior_beta_mu=prior_beta_mu,
                        prior_beta_cov=prior_beta_cov,
                        row_vec=prior_beta_cov_rowvec,
                        row_id=prior_beta_cov_rowid,
                        beta_can=beta_can
                    )

            rv = plpy.execute(sqlstr)
            logprior = rv[0]['logprior']
            logposterior_can = loglikelihood + logprior
            
            ratio = np.exp(logposterior_can - logposterior_curr)
        
            if (np.random.rand() < ratio):
                beta = beta_can
                logposterior_curr = logposterior_can
                num_accepts = num_accepts+1

            if (ite >= num_burnin_iter):
                sqlstr = \"""
                    insert into mcmc.mhtrace values ({ite},array{beta});
                \""".format(ite=ite,beta=beta)
                plpy.execute(sqlstr)
            
        sqlstr = \"""
            Metropolis Hastings coeffieicent trace stored in table mcmc.mhtrace. 
            Number of accepts = {num_accepts}
        \""".format(num_accepts=num_accepts)
        return sqlstr
    $$ language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """
    drop function if exists mcmc.MHLogit(
        varchar, 
        varchar,
        varchar,
        varchar,
        varchar,
        varchar,
        float8[], 
        varchar,
        varchar,
        varchar,
        int, 
        int
    );
"""
psql.execute(sql,conn)
conn.commit();

# Metropolis Hastings Logistic Regression Driver Function
# Arguments:
# x - name of the table storing the data matrix
# x_rowid - the row identifier column in the x data matrix
# x_rowvec - the vector per row in the x data matrix stored as an array
# y - name of the table storing the response column vector (binary column, should be 1 and 0)
# y_rowid : the row identifier for each element of the y vector. y_rowid should match x_rowid
# y_element: a 1-Dimensional array containing the y vector values
# prior_beta_mu : the mean of the normal beta prior distribution supplied as an array
# prior_beta_cov : name of the table storing the covariance matrix of the normal beta prior distribution
# prior_beta_cov_rowid : the row identifier column in the prior_beta_cov table
# prior_beta_cov_rowvec : the vector per row in the prior_beta_cov matrix stored as an array 
# num : number of Gibbs iterations
# num_burnin_iter: number of burnin iterations

sql = """
    create or replace function mcmc.MHLogit(
        x varchar,
        x_rowid varchar,
        x_rowvec varchar,
        y varchar,
        y_rowid varchar,
        y_element varchar,
        prior_beta_mu float8[],
        prior_beta_cov varchar,
        prior_beta_cov_rowid varchar,
        prior_beta_cov_rowvec varchar,
        num_iter int,
        num_burnin_iter int
    )
    returns text
    AS
    $$
        import numpy as np
        from scipy.stats import multivariate_normal as scipymvn
        from numpy.random import multivariate_normal as npmvn
                
        sqlstr = \"""
            select
                mcmc.array_agg_array({val} order by {rowid}) as cov,
                max(array_upper({val}, 1)) as ncols
            from            
                {covtable};
        \""".format(
                val=prior_beta_cov_rowvec, 
                rowid=prior_beta_cov_rowid,
                covtable=prior_beta_cov
            )
        rv = plpy.execute(sqlstr)
        cov = rv[0]['cov']
        d = rv[0]['ncols']
        # Reshape the cov linear array into a matrix
        cov = np.matrix(cov).reshape(len(cov)/d, d)
        plpy.info(cov.shape)
        invcovmat = np.linalg.inv(cov)
        
        # Compute x_t_x. This will be in memory. Should be good as long as x_t_x < 1 GB.
        # For example, for 10,000-8 byte floating point features, the size of x_t_x would
        # be 10,000 x 10,000 ~= 800 MB
        
        sqlstr = \"""
            select
                madlib.crossprod(
                    {xval},
                    {xval}
                ) as xtx
            from
                {x};
        \""".format(xval=x_rowvec,x=x)
        rv = plpy.execute(sqlstr)   
        xtx = rv[0]['xtx']
        # Reshape the xtx linear array into a matrix
        xtx = np.matrix(xtx).reshape(len(xtx)/d, d)
        plpy.info(xtx.shape)
        
        b_tilda = np.linalg.inv(xtx + invcovmat)
        num_features = xtx.shape[0]
        
        # Compute the OLS estimate as starting value for beta
        sqlstr = \"""
            select 
                madlib.crossprod(t1.{xval},t2.{yval}) as xty 
            from 
                {x} t1, 
                {y} t2 
            where t1.{x_rowid} = t2.{y_rowid};
        \""".format(x=x,y=y,xval=x_rowvec,yval=y_element,x_rowid=x_rowid,y_rowid=y_rowid)
        rv = plpy.execute(sqlstr)   
        xty = rv[0]['xty']
        xty = np.transpose(np.matrix(xty))
        beta = np.linalg.inv(xtx)*xty
        beta = (beta.flatten().tolist())[0]
        plpy.info(beta)
        
        plpy.execute('drop table if exists mcmc.x_beta;')
        sqlstr = \"""
            create table mcmc.x_beta as (
            select
                t1.{x_rowid},
                madlib.array_dot(t1.{xval}::float8[],array{beta}::float8[]) as xb
            from 
                {x} t1
            ) distributed by ({x_rowid});
        \""".format(x=x,xval=x_rowvec,x_rowid=x_rowid,beta=beta)
        plpy.execute(sqlstr)   
        
        sqlstr = \"""
            select 
                sum(ll) as loglikelihood 
            from (
                select
                    ((y*log(exp(1.0)::numeric,p::numeric)) + 
                        ((1.0-y)*log(exp(1.0)::numeric,(1.0-p)::numeric))) as ll
                from (
                    select
                        unnest(t1.{y_element}) as y,
                        1.0/(1.0 + exp(-1.0*t2.xb)) as p
                    from
                        {y_vector} t1,
                        mcmc.x_beta t2
                    where
                        t1.{y_rowid} = t2.{x_rowid}
                ) q1
            )q2;
        \""".format(
            y_element = y_element,
            y_vector = y,
            y_rowid = y_rowid,
            x_rowid = x_rowid
        )

        rv = plpy.execute(sqlstr)
        loglikelihood = rv[0]['loglikelihood']
        logprior = scipymvn.logpdf(x=beta,mean=prior_beta_mu,cov=cov)
        logposterior_curr = loglikelihood + logprior
        
        plpy.info(loglikelihood)
        plpy.info(logprior)
        plpy.info(logposterior_curr)

        num_accepts = 0
        coeffs = np.zeros(shape=(num_iter-num_burnin_iter,num_features))
        coefindx = 0

        plpy.execute('drop table if exists mcmc.mhtrace;')
        plpy.execute(\"""
            create table mcmc.mhtrace(iter_num int, beta float8[]) 
            distributed by (iter_num);
        \""")
        
        plpy.info('Entering Metropolis Hastings Iterations')
        
        insert_iter = 0
        insertsql = ''
        insertfreq = 1000
        
        for ite in range(0,num_iter):
            if (ite%100 == 0):
                plpy.info('Iteration number: {ite}'.format(ite=ite))
            beta_can = npmvn(mean=beta,cov=b_tilda,size =1).flatten().tolist() # This will be in memory. Should be ok
            # as this has dimensions = number features.

            plpy.execute('drop table if exists mcmc.x_beta;')
            sqlstr = \"""
                create table mcmc.x_beta as (
                    select
                        t1.{x_rowid},
                        madlib.array_dot(t1.{xval}::float8[],array{beta}::float8[]) as xb
                    from 
                        {x} t1
                ) distributed by ({x_rowid});
            \""".format(x=x,xval=x_rowvec,x_rowid=x_rowid,beta=beta_can)
            plpy.execute(sqlstr)

            sqlstr = \"""
                select 
                    sum(ll) as loglikelihood 
                from (
                    select
                        ((y*log(exp(1.0)::numeric,p::numeric)) + 
                            ((1.0-y)*log(exp(1.0)::numeric,(1.0-p)::numeric))) as ll
                    from (
                        select
                            unnest(t1.{y_element}) as y,
                            1.0/(1.0 + exp(-1.0*t2.xb)) as p
                        from
                            {y_vector} t1,
                            mcmc.x_beta t2
                        where
                            t1.{y_rowid} = t2.{x_rowid}
                    ) q1
                )q2;
            \""".format(
                y_element = y_element,
                y_vector = y,
                y_rowid = y_rowid,
                x_rowid = x_rowid
            )

            rv = plpy.execute(sqlstr)
            loglikelihood = rv[0]['loglikelihood']
            logprior = scipymvn.logpdf(x=beta_can,mean=prior_beta_mu,cov=cov)
            logposterior_can = loglikelihood + logprior
            
            ratio = np.exp(logposterior_can - logposterior_curr)
        
            if (np.random.rand() < ratio):
                beta = beta_can
                logposterior_curr = logposterior_can
                num_accepts = num_accepts+1

            if (ite >= num_burnin_iter):
                if (insert_iter == 0):
                    insertsql = 'insert into mcmc.mhtrace values ({ite},array{beta})'.format(ite=ite,beta=beta)
                    insert_iter = insert_iter + 1
                else:
                    insertsql = insertsql + ',({ite},array{beta})'.format(ite=ite,beta=beta)
                    insert_iter = insert_iter + 1
            
            if (insert_iter%insertfreq == 0):
                insert_iter = 0
                insertsql = insertsql + ';'
                plpy.execute(insertsql)
        if (insert_iter%insertfreq != 0):
            insertsql = insertsql + ';'
            plpy.execute(insertsql)
        sqlstr = \"""
            Metropolis Hastings coeffieicent trace stored in table mcmc.mhtrace. 
            Number of accepts = {num_accepts}
        \""".format(num_accepts=num_accepts)
        return sqlstr
    $$ language plpythonu;
"""
psql.execute(sql,conn)
conn.commit()

sql = """drop table if exists mcmc.lung_cancer_data;"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create table mcmc.lung_cancer_data (
        tumorsize float8,
        co2 float8,
        pain float8,
        wound float8,
        mobility float8,
        ntumors float8,
        nmorphine float8,
        lungcapacity float8,
        Age float8,
        Married int,
        Sex varchar,
        WBC float8,
        RBC float8,
        BMI float8,
        IL6 float8,
        CRP float8,
        remission int
    ) distributed randomly;
"""
psql.execute(sql,conn)
conn.commit()

psql.read_sql('select count(*) from mcmc.lung_cancer_data;',conn)

sql = """drop table if exists mcmc.mh_lung_cancer_data;"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create table mcmc.mh_lung_cancer_data as (
        select
            row_number() over() as rowid,
            *
        from
            mcmc.lung_cancer_data
        
    ) distributed randomly;
"""
psql.execute(sql,conn)
conn.commit()

psql.read_sql('select count(*) from mcmc.mh_lung_cancer_data;',conn)

sql = """drop table if exists mcmc.x;"""
psql.execute(sql,conn)
conn.commit()

sql = """drop table if exists mcmc.y;"""
psql.execute(sql,conn)
conn.commit()

sql = """drop table if exists mcmc.prior_beta_cov;"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create table mcmc.x as (
        select
            rowid,
            array[
                1,
                (tumorsize - (avg(tumorsize) OVER()))/(stddev(tumorsize) OVER()),
                (co2 - (avg(co2) OVER()))/(stddev(co2) OVER()), 
                (pain - (avg(pain) OVER()))/(stddev(pain) OVER()), 
                (wound - (avg(wound) OVER()))/(stddev(wound) OVER()), 
                (mobility - (avg(mobility) OVER()))/(stddev(mobility) OVER()),  
                (ntumors - (avg(ntumors) OVER()))/(stddev(ntumors) OVER()), 
                (nmorphine - (avg(nmorphine) OVER()))/(stddev(nmorphine) OVER()), 
                (lungcapacity - (avg(lungcapacity) OVER()))/(stddev(lungcapacity) OVER()), 
                (age - (avg(age) OVER()))/(stddev(age) OVER()), 
                (BMI - (avg(BMI) OVER()))/(stddev(BMI) OVER())
            ]::float8[] as row_vec
        from
            mcmc.mh_lung_cancer_data
        
    ) distributed by (rowid);
"""
psql.execute(sql,conn)
conn.commit()

sql = """
    create table mcmc.y as (
        select
            rowid,
            array[
                remission
            ]::float8[] as element
        from
            mcmc.mh_lung_cancer_data
        
    ) distributed by (rowid);
"""

psql.execute(sql,conn)
conn.commit()

sql = """
    create table mcmc.prior_beta_cov as (
        select 
            row_number() over() as row_id,
            row_vec
        from (
            select
                mcmc.create_identity_matrix(11) as row_vec
        )q
    ) distributed by (row_id);
"""
psql.execute(sql,conn)
conn.commit()

sql = """
    select 
        mcmc.MHLogit(
        'mcmc.x',
        'rowid', 
        'row_vec', 
        'mcmc.y', 
        'rowid', 
        'element', 
        array[0,0,0,0,0,0,0,0,0,0,0]::float8[], 
        'mcmc.prior_beta_cov', 
        'row_id',
        'row_vec',
        3000,
        300);
"""
psql.execute(sql,conn)
conn.commit()

# Let us now retrieve the coefficient trace
sql = """select iter_num, beta from mcmc.mhtrace order by iter_num;"""
df = psql.read_sql(sql,conn)
coeffs = np.array(df.beta.tolist())

# Plot the coefficients
f, axarr = plt.subplots(6,2)
f.patch.set_facecolor('white')
f.set_size_inches(12,20)
axarr[0,0].plot(range(0,coeffs.shape[0]),coeffs[:,0])
axarr[0,0].set_title('Intercept')
axarr[0,1].plot(range(0,coeffs.shape[0]),coeffs[:,1])
axarr[0,1].set_title('Coefficient of tumor size')
axarr[1,0].plot(range(0,coeffs.shape[0]),coeffs[:,2])
axarr[1,0].set_title('Coefficient of co2')
axarr[1,1].plot(range(0,coeffs.shape[0]),coeffs[:,3])
axarr[1,1].set_title('Coefficient of pain')
axarr[2,0].plot(range(0,coeffs.shape[0]),coeffs[:,4])
axarr[2,0].set_title('Coefficient of wound')
axarr[2,1].plot(range(0,coeffs.shape[0]),coeffs[:,5])
axarr[2,1].set_title('Coefficient of mobility')
axarr[3,0].plot(range(0,coeffs.shape[0]),coeffs[:,6])
axarr[3,0].set_title('Coefficient of ntumors')
axarr[3,1].plot(range(0,coeffs.shape[0]),coeffs[:,7])
axarr[3,1].set_title('Coefficient of nmorphine')
axarr[4,0].plot(range(0,coeffs.shape[0]),coeffs[:,8])
axarr[4,0].set_title('Coefficient of lung capacity')
axarr[4,1].plot(range(0,coeffs.shape[0]),coeffs[:,9])
axarr[4,1].set_title('Coefficient of age')
axarr[5,0].plot(range(0,coeffs.shape[0]),coeffs[:,10])
axarr[5,0].set_title('Coefficient of BMI')
axarr[5,1].axis('off')
plt.show()

# Plot the coefficient density
# Let us inspect the coefficients now
f, axarr = plt.subplots(6,2)
f.patch.set_facecolor('white')
f.set_size_inches(12,20)
sns.kdeplot(coeffs[:,0], ax=axarr[0,0])
axarr[0,0].set_title('Intercept')
sns.kdeplot(coeffs[:,1],ax=axarr[0,1])
axarr[0,1].set_title('Coefficient of tumor size')
sns.kdeplot(coeffs[:,2], ax=axarr[1,0])
axarr[1,0].set_title('Coefficient of co2')
sns.kdeplot(coeffs[:,3],ax=axarr[1,1])
axarr[1,1].set_title('Coefficient of pain')
sns.kdeplot(coeffs[:,4],ax=axarr[2,0])
axarr[2,0].set_title('Coefficient of wound')
sns.kdeplot(coeffs[:,5],ax=axarr[2,1])
axarr[2,1].set_title('Coefficient of mobility')
sns.kdeplot(coeffs[:,6],ax=axarr[3,0])
axarr[3,0].set_title('Coefficient of ntumors')
sns.kdeplot(coeffs[:,7],ax=axarr[3,1])
axarr[3,1].set_title('Coefficient of nmorphine')
sns.kdeplot(coeffs[:,8],ax=axarr[4,0])
axarr[4,0].set_title('Coefficient of lung capacity')
sns.kdeplot(coeffs[:,9],ax=axarr[4,1])
axarr[4,1].set_title('Coefficient of age')
sns.kdeplot(coeffs[:,10],ax=axarr[5,0])
axarr[5,0].set_title('Coefficient of BMI')
axarr[5,1].axis('off')
plt.show()

# What are the mean and median coeffs
beta_mean = np.mean(coeffs,axis=0)
beta_median = np.median(coeffs,axis=0)
print beta_mean
print beta_median

# Close DB connection
conn.close()



