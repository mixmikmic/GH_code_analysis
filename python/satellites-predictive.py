get_ipython().run_line_magic('load_ext', 'jupyter_probcomp.magics')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('vizgpm', 'inline')

get_ipython().system('rm -f resources/satellites.bdb')
get_ipython().run_line_magic('bayesdb', '-j resources/satellites.bdb')

get_ipython().run_line_magic('mml', "CREATE TABLE satellites_t FROM 'resources/satellites.csv';")
get_ipython().run_line_magic('mml', ".nullify satellites_t 'NaN'")

get_ipython().run_line_magic('sql', 'SELECT * FROM satellites_t ORDER BY RANDOM() LIMIT 5;')

get_ipython().run_line_magic('mml', '.guess_schema satellites_t')

get_ipython().run_line_magic('mml', 'CREATE POPULATION satellites FOR satellites_t WITH SCHEMA (GUESS STATTYPES FOR (*));')

get_ipython().run_line_magic('bql', '.interactive_pairplot --population=satellites    SELECT apogee_km, perigee_km, launch_mass_kg, dry_mass_kg, power_watts, class_of_orbit FROM satellites_t')

get_ipython().run_line_magic('mml', 'CREATE ANALYSIS SCHEMA satellites_crosscat FOR satellites WITH BASELINE crosscat;')

get_ipython().run_line_magic('mml', 'INITIALIZE 8 ANALYSES IF NOT EXISTS FOR satellites_crosscat;')

get_ipython().run_line_magic('mml', 'ANALYZE satellites_crosscat FOR 180 SECONDS WAIT (OPTIMIZED);')

get_ipython().run_line_magic('mml', '.render_crosscat --subsample=50 satellites_crosscat 0')
get_ipython().run_line_magic('mml', '.render_crosscat --subsample=50 satellites_crosscat 3')
get_ipython().run_line_magic('mml', '.render_crosscat --subsample=50 satellites_crosscat 5')
get_ipython().run_line_magic('mml', '.render_crosscat --subsample=50 satellites_crosscat 7')

get_ipython().run_line_magic('bql', '.interactive_heatmap ESTIMATE DEPENDENCE PROBABILITY FROM PAIRWISE VARIABLES OF satellites;')

get_ipython().run_line_magic('bql', '.interactive_heatmap ESTIMATE CORRELATION FROM PAIRWISE VARIABLES OF satellites WHERE CORRELATION PVALUE < 0.01;')

get_ipython().run_cell_magic('bql', '', "CREATE TEMP TABLE country_purpose_top10 AS\nSELECT\n    country_purpose,\n    COUNT(country_purpose) AS count\nFROM (\n    SELECT country_of_operator ||'--'|| purpose AS country_purpose\n    FROM satellites_t)\nGROUP BY country_purpose\nORDER BY count DESC\nLIMIT 10")

get_ipython().run_line_magic('bql', 'SELECT * FROM country_purpose_top10')

get_ipython().run_cell_magic('bql', '', "CREATE TEMP TABLE selections AS\nSELECT country_of_operator || '--' || purpose AS country_purpose\nFROM satellites_t\nWHERE country_purpose IN (\n    SELECT country_purpose FROM country_purpose_top10\n);")

get_ipython().run_cell_magic('bql', '', "CREATE TEMP TABLE simulations AS\nSELECT country_of_operator || '--' || purpose AS country_purpose\nFROM (\n    SIMULATE\n        country_of_operator,\n        purpose\n    FROM satellites\n    LIMIT 1000\n)\nWHERE country_purpose IN (\n    SELECT country_purpose FROM country_purpose_top10\n);")

get_ipython().run_cell_magic('sql', '', ".histogram_nominal --normed=True\nSELECT\n    country_purpose, 'selections'\nFROM selections\nUNION ALL\nSELECT\n    country_purpose, 'simulations'\nFROM simulations;")

get_ipython().run_line_magic('bql', '.density --xmin=-20000 --xmax=60000 SELECT perigee_km FROM satellites_t;')

get_ipython().run_line_magic('bql', '.density --xmin=-20000 --xmax=60000 SIMULATE perigee_km FROM satellites LIMIT 250;')

get_ipython().run_line_magic('bql', '.density --xmin=-250 --xmax=5000 SELECT period_minutes FROM satellites_t;')

get_ipython().run_line_magic('bql', '.density --xmin=-250 --xmax=5000 SIMULATE period_minutes FROM satellites LIMIT 250;')

get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SELECT period_minutes, perigee_km, class_of_orbit FROM satellites_t;')

get_ipython().run_line_magic('multiprocess', 'off')

get_ipython().run_cell_magic('bql', '', 'ESTIMATE\n    "name",\n    "apogee_km",\n    "perigee_km",\n    "class_of_orbit",\n    "period_minutes"\nFROM satellites\nWHERE (\n    PREDICTIVE PROBABILITY OF period_minutes\n    > PREDICTIVE PROBABILITY OF period_minutes GIVEN ("apogee_km", "perigee_km")\n)')

get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SELECT period_minutes, perigee_km, class_of_orbit FROM satellites_t;')

get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SIMULATE period_minutes, perigee_km, class_of_orbit FROM satellites LIMIT 250;')

get_ipython().run_cell_magic('venturescript', '', '// Kepler CGPM.\ndefine kepler = () -> {\n\n  // Kepler\'s law.\n  assume keplers_law = (apogee, perigee) -> {\n    let GM = 398600.4418;\n    let earth_radius = 6378;\n    let a = (abs(apogee) + abs(perigee)) *\n        0.5 + earth_radius;\n    2 * 3.14159265359 * sqrt(a**3 / GM) / 60\n  };\n\n  // Internal samplers.\n  assume outlier_prob = beta(2,5);\n  assume error_sampler = mem((is_outlier) -> {\n    make_nig_normal(1, 1, 1, 1)\n  });\n\n  // Output simulators.\n  assume sim_is_outlier = mem((rowid, apogee, perigee) ~> {\n      flip(outlier_prob) #latent:rowid\n  });\n\n  assume sim_period_error = mem((rowid, apogee, perigee) ~> {\n      let is_outlier = sim_is_outlier(rowid, apogee, perigee);\n      error_sampler(is_outlier)() #latent:rowid\n  });\n\n  assume sim_period = mem((rowid, apogee, perigee) ~> {\n      keplers_law(apogee, perigee) + sim_period_error(rowid, apogee, perigee)\n  });\n\n  // List of simulators.\n  assume simulators = [sim_period, sim_is_outlier, sim_period_error];\n};\n\n// Output observers.\ndefine obs_is_outlier = (rowid, apogee, perigee, value, label) -> {\n    $label: observe sim_is_outlier($rowid, $apogee, $perigee) = atom(value);\n};\ndefine obs_period_error = (rowid, apogee, perigee, value, label) -> {\n    $label: observe sim_period_error( $rowid, $apogee, $perigee) = value;\n};\ndefine obs_period = (rowid, apogee, perigee, value, label) -> {\n    let theoretical_period = run(sample keplers_law($apogee, $perigee));\n    obs_period_error(rowid, apogee, perigee, value - theoretical_period, label);\n};\n\n// List of observers.\ndefine observers = [obs_period, obs_is_outlier, obs_period_error];\n\n// List of inputs.\ndefine inputs = ["apogee", "perigee"];\n\n// Transition operator.\ndefine transition = (N) -> {mh(default, one, N)};')

get_ipython().run_cell_magic('mml', '', 'CREATE ANALYSIS SCHEMA satellites_crosscat_kepler FOR satellites WITH BASELINE crosscat (\n\n    OVERRIDE GENERATIVE MODEL\n    FOR\n        period_minutes\n    GIVEN\n        apogee_km, perigee_km\n    AND EXPOSE\n        kepler_cluster CATEGORICAL,\n        kepler_residual NUMERICAL\n    USING\n    venturescript(mode=venture_script, sp=kepler);\n\n    SUBSAMPLE 250\n)')

get_ipython().run_line_magic('mml', 'INITIALIZE 1 ANALYSIS FOR satellites_crosscat_kepler')

get_ipython().run_cell_magic('mml', '', 'ANALYZE satellites_crosscat_kepler FOR 750 ITERATIONS WAIT (\n    VARIABLES period_minutes, kepler_cluster, kepler_residual;\n)')

get_ipython().run_cell_magic('mml', '', 'ANALYZE satellites_crosscat_kepler FOR 1 MINUTE WAIT (\n    SKIP period_minutes, kepler_cluster, kepler_residual;\n    OPTIMIZED\n)')

get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SELECT period_minutes, perigee_km, class_of_orbit     FROM satellites_t;')
    
get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SIMULATE period_minutes, perigee_km, class_of_orbit     FROM satellites MODELED BY satellites_crosscat LIMIT 250;')

get_ipython().run_line_magic('bql', '.scatter --ymin=1e1 --ymax=1e6 --xmin=1e1 --xmax=1e6 --xlog=10 --ylog=10     SIMULATE period_minutes, perigee_km, class_of_orbit     FROM satellites MODELED BY satellites_crosscat_kepler LIMIT 250;')

