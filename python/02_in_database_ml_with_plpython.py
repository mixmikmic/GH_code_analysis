get_ipython().run_line_magic('run', "'00_database_connectivity_setup.ipynb'")
IPython.display.clear_output()

get_ipython().run_cell_magic('time', '', '%%bash\nmkdir -p /tmp/postgresopen_2017\nif [ ! -f /tmp/postgresopen_2017/winequality-red.csv ]; then\n    echo "Fetching UCI Wine Quality dataset"\n    curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv > /tmp/postgresopen_2017/winequality-red.csv\nfi\nls -l /tmp/postgresopen_2017/winequality-red.csv')

get_ipython().run_cell_magic('time', '', "%%execsql\ndrop table if exists wine_sample;\ncreate table wine_sample\n(\n    id serial,\n    fixed_acidity float8,\n    volatile_acidity float8,\n    citric_acid float8,\n    residual_sugar float8,\n    chlorides  float8,\n    free_sulfur_dioxide float8,\n    total_sulfur_dioxide float8,\n    density float8,\n    ph float8,\n    sulphates float8,\n    alcohol float8,\n    quality float8\n);\ncopy \nwine_sample\n(\n    fixed_acidity,\n    volatile_acidity,\n    citric_acid,\n    residual_sugar,\n    chlorides,\n    free_sulfur_dioxide,\n    total_sulfur_dioxide,\n    density,\n    ph,\n    sulphates,\n    alcohol,\n    quality\n) from '/tmp/postgresopen_2017/winequality-red.csv' WITH DELIMITER ';' CSV HEADER;")

get_ipython().run_cell_magic('time', '', '%%showsql\nselect\n    *\nfrom\n    wine_sample\nlimit 10;')

get_ipython().run_cell_magic('time', '', '%%showsql\nselect\n    id,\n    ARRAY[\n        fixed_acidity,\n        volatile_acidity,\n        citric_acid,\n        residual_sugar,\n        chlorides,\n        free_sulfur_dioxide,\n        total_sulfur_dioxide,\n        density,\n        ph,\n        sulphates,\n        alcohol\n    ] as features,\n    quality\nfrom\n    wine_sample\nlimit 10;')

get_ipython().run_cell_magic('time', '', "%%execsql\n--1) Define model return type record\ndrop type if exists host_mdl_coef_intercept CASCADE;\ncreate type host_mdl_coef_intercept\nAS\n(\n    hostname text, -- hostname on which the model was built\n    coef float[], -- model coefficients\n    intercept float, -- intercepts\n    r_square float -- training data fit\n);\n\n--2) Define a UDA to concatenate arrays\ndrop aggregate if exists array_agg_array(anyarray) CASCADE;\ncreate aggregate array_agg_array(anyarray)\n(\n    SFUNC = array_cat,\n    STYPE = anyarray\n);\n\n--3) Define PL/Python function to train ridge regression model\ncreate or replace function sklearn_ridge_regression(\n    features_mat float8[],\n    n_features int,\n    labels float8[]\n)\nreturns host_mdl_coef_intercept\nas\n$$\n    import os\n    from sklearn import linear_model, preprocessing\n    import numpy as np\n    X_unscaled = np.array(features_mat).reshape(int(len(features_mat)/n_features), int(n_features))\n    # Scale the input (zero mean, unit variance)\n    X = preprocessing.scale(X_unscaled)\n    y = np.array(labels).transpose()\n    mdl = linear_model.Ridge(alpha = .5)\n    mdl.fit(X, y)\n    result = [\n        os.popen('hostname').read().strip(), \n        mdl.coef_, \n        mdl.intercept_, \n        mdl.score(X, y)\n    ] \n    return result\n$$language plpython3u;")

get_ipython().run_cell_magic('showsql', '', 'select\n    (\n        sklearn_ridge_regression(\n            features_mat,\n            n_features,\n            labels\n        )\n    ).*\nfrom\n(\n    select\n        -- Convert rows of features into a large linear array\n        array_agg_array(features order by id) as features_mat,\n        -- Number of features\n        max(array_upper(features, 1)) as n_features,\n        -- Gather all the Labels\n        array_agg(quality order by id) as labels\n    from\n    (\n        select\n            id,\n            1 as grouping,\n            -- Create a feature vector of independent variables\n            ARRAY[\n                fixed_acidity,\n                volatile_acidity,\n                citric_acid,\n                residual_sugar,\n                chlorides,\n                free_sulfur_dioxide,\n                total_sulfur_dioxide,\n                density,\n                ph,\n                sulphates,\n                alcohol\n            ] as features,\n            quality\n        from\n            wine_sample\n    )q1\n    group by\n        grouping\n)q2')

get_ipython().run_cell_magic('time', '', "%%execsql\n-- An array of 120000000 float8(8 bytes) types = 960 MB\n--1) Define UDF to generate large arrays\ncreate or replace function gen_array(x int)\nreturns float8[]\nas\n$$\n    from random import random\n    return [random() for _ in range(x)]\n$$language plpython3u;\n\n--2) Create a table\ndrop table if exists cellsize_test;\ncreate table cellsize_test\nas\n(\n    select\n        1 as row,\n        y,\n        gen_array(12000) as arr\n    from\n        generate_series(1, 3) y\n);\n\n--3) Define a UDA to concatenate arrays\nDROP AGGREGATE IF EXISTS array_agg_array(anyarray) CASCADE;\nCREATE AGGREGATE array_agg_array(anyarray)\n(\n    SFUNC = array_cat,\n    STYPE = anyarray\n);\n\n--4) Define a UDF to consume a really large array and return its size\ncreate or replace function consume_large_array(x float8[])\nreturns text\nas\n$$\n    return 'size of x:{0}'.format(len(x))\n$$language plpython3u;")

get_ipython().run_cell_magic('time', '', '%%showsql\n--5) Invoke the UDF & UDA to demonstrate failure due to max_fieldsize_limit\nselect\n    row,\n    consume_large_array(arr)\nfrom\n(\n\n    select\n        row,\n        array_agg_array(arr) as arr\n    from\n        cellsize_test\n    group by\n        row\n)q;')

get_ipython().run_cell_magic('time', '', "%%execsql\n--1) SFUNC: State transition function, part of a User-Defined-Aggregate definition\n-- This function will merely stack every row of input, into the GD variable\ndrop function if exists stack_rows(\n    text,\n    text[],\n    float8[],\n    float8\n) cascade;\n\ncreate or replace function stack_rows(\n    key text,\n    header text[], -- name of the features column and the dependent variable column\n    features float8[], -- independent variables (as array)\n    label float8 -- dependent variable column\n)\nreturns text\nas\n$$\n    if 'header' not in GD:\n        GD['header'] = header\n    if not key:\n        gd_key = 'stack_rows'\n        GD[gd_key] = [[features, label]]\n        return gd_key\n    else:\n        GD[key].append([features, label])\n        return key\n$$language plpython3u;\n\n--2) Define the User-Defined Aggregate (UDA) consisting of a state-transition function (SFUNC), a state variable and a FINALFUNC (optional)\ndrop aggregate if exists stack_rows( \n    text[], -- header (feature names)\n    float8[], -- features (feature values),\n    float8 -- labels\n) cascade;\ncreate aggregate stack_rows(\n        text[], -- header (feature names)\n        float8[], -- features (feature values),\n        float8 -- labels\n    )\n(\n    SFUNC = stack_rows,\n    STYPE = text -- the key in GD used to hold the data across calls\n);\n\n--3) Create a return type for model results\nDROP TYPE IF EXISTS host_mdl_coef_intercept CASCADE;\nCREATE TYPE host_mdl_coef_intercept\nAS\n(\n    hostname text, -- hostname on which the model was built\n    coef float[], -- model coefficients\n    intercept float, -- intercepts\n    r_square float -- training data fit\n);\n\n--4) Define a UDF to run ridge regression by retrieving the data from the key in GD and returning results\ndrop function if exists run_ridge_regression(text) cascade;\ncreate or replace function run_ridge_regression(key text)\nreturns host_mdl_coef_intercept\nas\n$$\n    import os\n    import numpy as np   \n    import pandas as pd\n    from sklearn import linear_model, preprocessing\n    \n    if key in GD:\n        df = pd.DataFrame(GD[key], columns=GD['header'])\n        mdl = linear_model.Ridge(alpha = .5)\n        X_unscaled = np.mat(df[GD['header'][0]].values.tolist())\n        # Scale the input (zero mean, unit variance)\n        X = preprocessing.scale(X_unscaled)\n        y = np.mat(df[GD['header'][1]].values.tolist()).transpose()\n        mdl.fit(X, y)\n        result = [\n            os.popen('hostname').read().strip(), \n            mdl.coef_[0], \n            mdl.intercept_[0], \n            mdl.score(X, y)\n        ]   \n        GD[key] = result        \n        result = GD[key]\n        del GD[key]\n        return result\n    else:\n        plpy.info('returning None')\n        return None\n$$ language plpython3u;")

get_ipython().run_cell_magic('time', '', "%%showsql\nselect\n    model,\n    (results).*\nfrom\n(\n    select\n        model,\n        run_ridge_regression(\n            stacked_input_key\n        ) as results\n    from\n    (\n        select\n            model,\n            stack_rows(\n                ARRAY['features', 'quality'], --header or names of input fields\n                features, -- feature vector input field\n                quality -- label column\n            ) as stacked_input_key\n        from\n        (\n            select\n                id,\n                1 as model,\n                -- Create a feature vector of independent variables\n                ARRAY[\n                    fixed_acidity,\n                    volatile_acidity,\n                    citric_acid,\n                    residual_sugar,\n                    chlorides,\n                    free_sulfur_dioxide,\n                    total_sulfur_dioxide,\n                    density,\n                    ph,\n                    sulphates,\n                    alcohol\n                ] as features,\n                quality\n            from\n                wine_sample\n        ) q1\n        group by\n            model\n    )q2\n)q3;")

