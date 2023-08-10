get_ipython().run_cell_magic('bash', '', "sqlplus -s test/test@dbserver:1521/orcl.mydomain.com <<EOF\n\nset serveroutput on\nbegin\n  dbms_output.put_line('Hello world!');\nend;\n/\n\nselect sysdate from dual;\n\nEOF")

get_ipython().run_cell_magic('bash', '', 'sqlplus -s scott/tiger@dbserver:1521/orcl.mydomain.com <<EOF\n\nset linesize 100\nset pagesize 100\ncol ename for a15\ncol job for a15\n\nSELECT * from EMP;\n\nEOF')

get_ipython().run_cell_magic('bash', '', 'sqlplus -s test/test@dbserver:1521/orcl.mydomain.com <<EOF\n\nset verify off\nset lines 4000\nset pages 999\nset heading off\n\n-- From https://github.com/LucaCanali/Miscellaneous/tree/master/SQL_color_Mandelbrot\n-- Configuration parameters for the Mandelbrot set calculation\n-- Edit to change the region displayed and/or resolution by changing the definitions here below\n-- Edit your terminal screen resolution and/or modify XPOINTS and YPOINTS so that the image fits the screen\ndefine XMIN=-2.0\ndefine YMIN=-1.4\ndefine XMAX=0.5\ndefine YMAX=1.4\ndefine XPOINTS=120\ndefine YPOINTS=60\ndefine XSTEP="(&XMAX - &XMIN)/(&XPOINTS - 1)"\ndefine YSTEP="(&YMAX - &YMIN)/(&YPOINTS - 1)"\n\n-- Visualization parameters \ndefine COLORMAP="012223333344445555666677770"\ndefine MAXITER="LENGTH(\'&COLORMAP\')"\ndefine BLUE_PALETTE="0,0,1,15,2,51,3,45,4,39,5,33,6,27,7,21"\ndefine PALETTE_NUMCOLS=8\ndefine ESCAPE_VAL=4\ndefine ANSICODE_PREFIX="chr(27)||\'[48;5;\'"\ndefine ANSICODE_BACKTONORMAL="chr(27)||\'[0m\'"\n\nWITH\n   XGEN AS (                            -- X dimension values generator\n        SELECT CAST(&XMIN + &XSTEP * (rownum-1) AS binary_double) AS X, rownum AS IX FROM DUAL CONNECT BY LEVEL <= &XPOINTS),\n   YGEN AS (                            -- Y dimension values generator\n        SELECT CAST(&YMIN + &YSTEP * (rownum-1) AS binary_double) AS Y, rownum AS IY FROM DUAL CONNECT BY LEVEL <= &YPOINTS),\n   Z(IX, IY, CX, CY, X, Y, I) AS (     -- Z point iterator. Makes use of recursive common table expression \n        SELECT IX, IY, X, Y, X, Y, 0 FROM XGEN, YGEN\n        UNION ALL\n        SELECT IX, IY, CX, CY, X*X - Y*Y + CX, 2*X*Y + CY, I+1 FROM Z WHERE X*X + Y*Y < &ESCAPE_VAL AND I < &MAXITER),\n   MANDELBROT_MAP AS (                       -- Computes an approximated map of the Mandelbrot set\n        SELECT IX, IY, MAX(I) AS VAL FROM Z  -- VAL=MAX(I) represents how quickly the values reached the escape point\n        GROUP BY IY, IX),\n   PALETTE AS (                              -- Color palette generator using ANSI escape codes\n        SELECT rownum-1 ID, &ANSICODE_PREFIX|| DECODE(rownum-1, &BLUE_PALETTE) || \'m \' || &ANSICODE_BACKTONORMAL COLOR \n        FROM DUAL CONNECT BY LEVEL <= &PALETTE_NUMCOLS)\nSELECT LISTAGG(PALETTE.COLOR) WITHIN GROUP (ORDER BY IX) GRAPH        -- The function LISTAGG concatenates values into rows\nFROM MANDELBROT_MAP, PALETTE\nWHERE TO_NUMBER(SUBSTR(\'&COLORMAP\',MANDELBROT_MAP.VAL,1))=PALETTE.ID  -- Map visualization using PALETTE and COLORMAP\nGROUP BY IY\nORDER BY IY DESC;\n\nEOF')


