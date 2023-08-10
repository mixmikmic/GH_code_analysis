get_ipython().magic('load_ext sql')

# %sql postgresql://gpdbchina@10.194.10.68:55000/madlib
get_ipython().magic('sql postgresql://fmcquillan@localhost:5432/madlib')

get_ipython().magic('sql select madlib.version();')

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS eventlog, path_output, path_output_tuples CASCADE;\nCREATE TABLE eventlog (event_timestamp TIMESTAMP,\n            user_id INT,\n            session_id INT,\n            page TEXT,\n            revenue FLOAT);\nINSERT INTO eventlog VALUES\n('04/15/2015 01:03:00', 100821, 100, 'LANDING', 0),\n('04/15/2015 01:04:00', 100821, 100, 'WINE', 0),\n('04/15/2015 01:05:00', 100821, 100, 'CHECKOUT', 39),\n('04/15/2015 02:06:00', 100821, 101, 'WINE', 0),\n('04/15/2015 02:09:00', 100821, 101, 'WINE', 0),\n('04/15/2015 01:15:00', 101121, 102, 'LANDING', 0),\n('04/15/2015 01:16:00', 101121, 102, 'WINE', 0),\n('04/15/2015 01:17:00', 101121, 102, 'CHECKOUT', 15),\n('04/15/2015 01:18:00', 101121, 102, 'LANDING', 0),\n('04/15/2015 01:19:00', 101121, 102, 'HELP', 0),\n('04/15/2015 01:21:00', 101121, 102, 'WINE', 0),\n('04/15/2015 01:22:00', 101121, 102, 'CHECKOUT', 23),\n('04/15/2015 02:15:00', 101331, 103, 'LANDING', 0),\n('04/15/2015 02:16:00', 101331, 103, 'WINE', 0),\n('04/15/2015 02:17:00', 101331, 103, 'HELP', 0),\n('04/15/2015 02:18:00', 101331, 103, 'WINE', 0),\n('04/15/2015 02:19:00', 101331, 103, 'CHECKOUT', 16),\n('04/15/2015 02:22:00', 101443, 104, 'BEER', 0),\n('04/15/2015 02:25:00', 101443, 104, 'CHECKOUT', 12),\n('04/15/2015 02:29:00', 101881, 105, 'LANDING', 0),\n('04/15/2015 02:30:00', 101881, 105, 'BEER', 0),\n('04/15/2015 01:05:00', 102201, 106, 'LANDING', 0),\n('04/15/2015 01:06:00', 102201, 106, 'HELP', 0),\n('04/15/2015 01:09:00', 102201, 106, 'LANDING', 0),\n('04/15/2015 02:15:00', 102201, 107, 'WINE', 0),\n('04/15/2015 02:16:00', 102201, 107, 'BEER', 0),\n('04/15/2015 02:17:00', 102201, 107, 'WINE', 0),\n('04/15/2015 02:18:00', 102871, 108, 'BEER', 0),\n('04/15/2015 02:19:00', 102871, 108, 'WINE', 0),\n('04/15/2015 02:22:00', 102871, 108, 'CHECKOUT', 21),\n('04/15/2015 02:25:00', 102871, 108, 'LANDING', 0),\n('04/15/2015 02:17:00', 103711, 109, 'BEER', 0),\n('04/15/2015 02:18:00', 103711, 109, 'LANDING', 0),\n('04/15/2015 02:19:00', 103711, 109, 'WINE', 0);\n\nSELECT * FROM eventlog ORDER BY event_timestamp ASC;")

get_ipython().run_cell_magic('sql', '', "SELECT madlib.path(\n     'eventlog',                -- Name of input table\n     'path_output',             -- Table name to store path results\n     'session_id',              -- Partition input table by session\n     'event_timestamp ASC',     -- Order partitions in input table by time\n     'buy:=page=''CHECKOUT''',  -- Define a symbol for checkout events\n     '(buy)',                   -- Pattern search: purchase\n     'sum(revenue) as checkout_rev',    -- Aggregate:  sum revenue by checkout\n     TRUE                       -- Persist matches\n     );\nSELECT * FROM path_output ORDER BY session_id, match_id;")

get_ipython().run_cell_magic('sql', '', 'SELECT session_id, sum(checkout_rev) FROM path_output GROUP BY session_id ORDER BY session_id;')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM path_output_tuples ORDER BY session_id ASC, event_timestamp ASC;')

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS path_output, path_output_tuples;\nSELECT madlib.path(\n     'eventlog',                -- Name of input table\n     'path_output',             -- Table name to store path results\n     'session_id',              -- Partition input table by session\n     'event_timestamp ASC',     -- Order partitions in input table by time\n      $$ land:=page='LANDING',\n        wine:=page='WINE',\n        beer:=page='BEER',\n        buy:=page='CHECKOUT',\n        other:=page<>'LANDING' AND page<>'WINE' AND page<>'BEER' AND  page<>'CHECKOUT'\n        $$,                     -- Symbols for page types\n\n      '(land)[^(land)(buy)]{0,2}(buy)', -- Purchase within 4 pages entering site\n     'sum(revenue) as checkout_rev',    -- Aggregate:  sum revenue by checkout\n     TRUE                       -- Persist matches\n     );\nSELECT * FROM path_output ORDER BY session_id, match_id;")

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM path_output_tuples ORDER BY session_id ASC, event_timestamp ASC;')

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS path_output, path_output_tuples;\nSELECT madlib.path(\n     'eventlog',                -- Name of input table\n     'path_output',             -- Table name to store path results\n     'session_id',              -- Partition input table by session\n     'event_timestamp ASC',     -- Order partitions in input table by time\n      $$ land:=page='LANDING',\n        wine:=page='WINE',\n        beer:=page='BEER',\n        buy:=page='CHECKOUT',\n        other:=page<>'LANDING' AND page<>'WINE' AND page<>'BEER' AND  page<>'CHECKOUT'\n        $$,                     -- Symbols for page types\n      '(land)[^(land)(buy)]{0,2}(buy)', -- Purchase within 4 pages entering site\n     '(max(event_timestamp)-min(event_timestamp)) as elapsed_time',    -- Aggregate: elapsed time\n     TRUE                       -- Persist matches\n     );\nSELECT * FROM path_output ORDER BY session_id, match_id;")

get_ipython().run_cell_magic('sql', '', "SELECT DATE(event_timestamp), user_id, session_id, revenue,\n    avg(revenue) OVER (PARTITION BY DATE(event_timestamp)) as avg_checkout_rev\n    FROM path_output_tuples\n    WHERE page='CHECKOUT'\n    ORDER BY user_id, session_id;")

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS path_output, path_output_tuples;\nSELECT madlib.path(\n     'eventlog',                -- Name of input table\n     'path_output',             -- Table name to store path results\n     'session_id',              -- Partition input table by session\n     'event_timestamp ASC',     -- Order partitions in input table by time\n      $$ land:=page='LANDING',\n        wine:=page='WINE',\n        beer:=page='BEER',\n        buy:=page='CHECKOUT',\n        other:=page<>'LANDING' AND page<>'WINE' AND page<>'BEER' AND  page<>'CHECKOUT'\n        $$,                     -- Symbols for page types\n      '[^(buy)](buy)',          -- Pattern to match\n     'array_agg(page ORDER BY session_id ASC, event_timestamp ASC) as page_path');\n     \nSELECT count(*), page_path from\n    (SELECT * FROM path_output) q\nGROUP BY page_path\nORDER BY count(*) DESC\nLIMIT 10;")

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS path_output, path_output_tuples;\nSELECT madlib.path(                                                                   \n     'eventlog',                    -- Name of the table                                           \n     'path_output',                 -- Table name to store the path results                         \n     'session_id',                  -- Partition by session                 \n     'event_timestamp ASC',         -- Order partitions in input table by time       \n     $$ nobuy:=page<>'CHECKOUT',\n        buy:=page='CHECKOUT'\n     $$,  -- Definition of symbols used in the pattern definition \n     '(nobuy)+(buy)',         -- At least one page followed by and ending with a CHECKOUT.\n     'array_agg(page ORDER BY session_id ASC, event_timestamp ASC) as page_path',  \n     FALSE,                        -- Don't persist matches\n     TRUE                          -- Turn on overlapping patterns\n     );\nSELECT * FROM path_output ORDER BY session_id, match_id;")

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS path_output, path_output_tuples;\nSELECT madlib.path(                                                                   \n     'eventlog',                    -- Name of the table                                           \n     'path_output',                 -- Table name to store the path results                         \n     'session_id',                  -- Partition by session                 \n     'event_timestamp ASC',         -- Order partitions in input table by time       \n     $$ nobuy:=page<>'CHECKOUT',\n        buy:=page='CHECKOUT'\n     $$,  -- Definition of symbols used in the pattern definition \n     '(nobuy)+(buy)',         -- At least one page followed by and ending with a CHECKOUT.\n     'array_agg(page ORDER BY session_id ASC, event_timestamp ASC) as page_path',  \n     FALSE,                        -- Don't persist matches\n     FALSE                          -- Turn on overlapping patterns\n     );\nSELECT * FROM path_output ORDER BY session_id, match_id;")
