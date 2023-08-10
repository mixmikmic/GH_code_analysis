get_ipython().magic('run db2.ipynb')

get_ipython().magic('sql -sampledata')

get_ipython().run_cell_magic('sql', '', 'SELECT DEPTNAME, LASTNAME FROM\n  DEPARTMENT D LEFT OUTER JOIN EMPLOYEE E\n  ON D.DEPTNO = E.WORKDEPT')

get_ipython().run_cell_magic('sql', '', 'SELECT DEPTNAME, LASTNAME FROM\n  DEPARTMENT D, EMPLOYEE E\nWHERE D.DEPTNO = E.WORKDEPT (+)')

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE LONGER_CHAR;\n  \nCREATE TABLE LONGER_CHAR\n  (\n  NAME CHAR(255)\n  );')

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE HEXEY;\n\nCREATE TABLE HEXEY\n  (\n  AUDIO_SHORT BINARY(255),\n  AUDIO_LONG  VARBINARY(1024),\n  AUDIO_CHAR  VARCHAR(255) FOR BIT DATA\n  );')

get_ipython().run_cell_magic('sql', '', "INSERT INTO HEXEY VALUES\n  (BINARY('Hello there'), \n   BX'2433A5D5C1', \n   VARCHAR_BIT_FORMAT(HEX('Hello there')));\n\nSELECT * FROM HEXEY;")

get_ipython().run_cell_magic('sql', '', 'UPDATE HEXEY \n  SET AUDIO_CHAR = AUDIO_SHORT')

get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*) FROM HEXEY WHERE\n  AUDIO_SHORT = AUDIO_CHAR')

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE TRUEFALSE;\n\nCREATE TABLE TRUEFALSE (\n    EXAMPLE INT,\n    STATE   BOOLEAN\n);')

get_ipython().run_cell_magic('sql', '', "INSERT INTO TRUEFALSE VALUES\n  (1, TRUE), \n  (2, FALSE),\n  (3, 0),\n  (4, 't'),\n  (5, 'no')")

get_ipython().magic('sql SELECT * FROM TRUEFALSE')

get_ipython().run_cell_magic('sql', '', "SELECT * FROM TRUEFALSE\n  WHERE STATE = TRUE OR STATE = 1 OR STATE = 'on' OR STATE IS TRUE")

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE SYNONYM_EMPLOYEE;\n\nCREATE TABLE SYNONYM_EMPLOYEE\n  (\n  NAME VARCHAR(20),\n  SALARY     INT4,\n  BONUS      INT2,\n  COMMISSION INT8,\n  COMMISSION_RATE FLOAT4,\n  BONUS_RATE FLOAT8\n  );')

get_ipython().run_cell_magic('sql', '', "SELECT DISTINCT(NAME), COLTYPE, LENGTH FROM SYSIBM.SYSCOLUMNS \n  WHERE TBNAME='SYNONYM_EMPLOYEE' AND TBCREATOR=CURRENT USER")

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE XYCOORDS;\n\nCREATE TABLE XYCOORDS\n  (\n  X INT,\n  Y INT\n  );\n    \nINSERT INTO XYCOORDS\n   WITH TEMP1(X) AS\n     (\n     VALUES (0)\n     UNION ALL\n     SELECT X+1 FROM TEMP1 WHERE X < 10\n     )\n   SELECT X, 2*X + 5\n     FROM TEMP1;')

get_ipython().run_cell_magic('sql', '', "SELECT 'COVAR_POP', COVAR_POP(X,Y) FROM XYCOORDS\nUNION ALL\nSELECT 'COVARIANCE', COVARIANCE(X,Y) FROM XYCOORDS")

get_ipython().run_cell_magic('sql', '', "SELECT 'STDDEV_POP', STDDEV_POP(X) FROM XYCOORDS\nUNION ALL\nSELECT 'STDDEV', STDDEV(X) FROM XYCOORDS")

get_ipython().run_cell_magic('sql', '', "SELECT 'VAR_SAMP', VAR_SAMP(X) FROM XYCOORDS\nUNION ALL\nSELECT 'VARIANCE_SAMP', VARIANCE_SAMP(X) FROM XYCOORDS")

get_ipython().run_cell_magic('sql', '', "WITH EMP(LASTNAME, WORKDEPT) AS\n  (\n  VALUES ('George','A01'),\n         ('Fred',NULL),\n         ('Katrina','B01'),\n         ('Bob',NULL)\n  )\nSELECT * FROM EMP WHERE \n   WORKDEPT ISNULL")

get_ipython().run_cell_magic('sql', '', "VALUES ('LOG',LOG(10))\nUNION ALL\nVALUES ('LN', LN(10))")

get_ipython().run_cell_magic('sql', '', "VALUES ('RANDOM', RANDOM())\nUNION ALL\nVALUES ('RAND', RAND())")

get_ipython().run_cell_magic('sql', '', "VALUES ('POSSTR',POSSTR('Hello There','There'))\nUNION ALL\nVALUES ('STRPOS',STRPOS('Hello There','There'))")

get_ipython().run_cell_magic('sql', '', "VALUES ('LEFT',LEFT('Hello There',5))\nUNION ALL\nVALUES ('STRLEFT',STRLEFT('Hello There',5))")

get_ipython().run_cell_magic('sql', '', "VALUES ('RIGHT',RIGHT('Hello There',5))\nUNION ALL\nVALUES ('STRRIGHT',STRRIGHT('Hello There',5))")

get_ipython().run_cell_magic('sql', '', "WITH SPECIAL(OP, DESCRIPTION, EXAMPLE, RESULT) AS\n  (\n  VALUES \n     (' | ','OR        ', '2 | 3   ', 2 | 3),\n     (' & ','AND       ', '2 & 3   ', 2 & 3),\n     (' ^ ','XOR       ', '2 ^ 3   ', 2 ^ 3),\n     (' ~ ','COMPLEMENT', '~2      ', ~2),\n     (' # ','NONE      ', '        ',0)\n  )\nSELECT * FROM SPECIAL")

get_ipython().run_cell_magic('sql', '', "SET SQL_COMPAT = 'NPS';\nWITH SPECIAL(OP, DESCRIPTION, EXAMPLE, RESULT) AS\n  (\n  VALUES \n     (' | ','OR        ', '2 | 3   ', 2 | 3),\n     (' & ','AND       ', '2 & 3   ', 2 & 3),\n     (' ^ ','POWER     ', '2 ^ 3   ', 2 ^ 3),\n     (' ~ ','COMPLIMENT', '~2      ', ~2),\n     (' # ','XOR       ', '2 # 3   ', 2 # 3)\n  )\nSELECT * FROM SPECIAL;")

get_ipython().run_cell_magic('sql', '', "SET SQL_COMPAT='DB2';\n\nSELECT WORKDEPT,INT(AVG(SALARY)) \n  FROM EMPLOYEE\nGROUP BY WORKDEPT;")

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, INT(AVG(SALARY))\n  FROM EMPLOYEE\nGROUP BY 1;')

get_ipython().run_cell_magic('sql', '', "SET SQL_COMPAT='NPS';\nSELECT WORKDEPT, INT(AVG(SALARY))\n  FROM EMPLOYEE\nGROUP BY 1;")

get_ipython().run_cell_magic('sql', '', "SET SQL_COMPAT = 'NPS';\nVALUES TRANSLATE('Hello');")

get_ipython().magic("sql VALUES TRANSLATE('Hello','o','1')")

get_ipython().magic("sql VALUES TRANSLATE('Hello','oe','12')")

get_ipython().magic("sql VALUES TRANSLATE('Hello','oel','12')")

get_ipython().magic("sql SET SQL_COMPAT='DB2'")

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME FROM EMPLOYEE\n  FETCH FIRST 5 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME FROM EMPLOYEE\n  ORDER BY LASTNAME\n  FETCH FIRST 5 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, COUNT(*) FROM EMPLOYEE\n  GROUP BY WORKDEPT\n  ORDER BY WORKDEPT')

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, COUNT(*) FROM EMPLOYEE\n  GROUP BY WORKDEPT\n  ORDER BY WORKDEPT\n  FETCH FIRST 5 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME FROM EMPLOYEE\n  FETCH FIRST 10 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME FROM EMPLOYEE\n  OFFSET 0 ROWS\n  FETCH FIRST 10 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME FROM EMPLOYEE\n  OFFSET 5 ROWS\n  FETCH FIRST 5 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, AVG(SALARY) FROM EMPLOYEE\nGROUP BY WORKDEPT\nORDER BY AVG(SALARY) DESC;')

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, AVG(SALARY) FROM EMPLOYEE\nGROUP BY WORKDEPT\nORDER BY AVG(SALARY) DESC\nOFFSET 2 ROWS FETCH FIRST 1 ROWS ONLY')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME, SALARY FROM EMPLOYEE\n  WHERE\n    SALARY > (\n       SELECT AVG(SALARY) FROM EMPLOYEE\n         GROUP BY WORKDEPT\n         ORDER BY AVG(SALARY) DESC\n         OFFSET 2 ROWS FETCH FIRST 1 ROW ONLY\n       )\nORDER BY SALARY')

get_ipython().run_cell_magic('sql', '', 'SELECT WORKDEPT, AVG(SALARY) FROM EMPLOYEE\nGROUP BY WORKDEPT\nORDER BY AVG(SALARY) DESC\nLIMIT 1 OFFSET 2')

get_ipython().run_cell_magic('sql', '', 'SELECT LASTNAME, SALARY FROM EMPLOYEE\n  WHERE\n    SALARY > (\n       SELECT AVG(SALARY) FROM EMPLOYEE\n         GROUP BY WORKDEPT\n         ORDER BY AVG(SALARY) DESC\n         LIMIT 2,1 \n       )\nORDER BY SALARY')

get_ipython().run_cell_magic('sql', '-q', 'DROP VARIABLE XINT2; \nDROP VARIABLE YINT2;\nDROP VARIABLE XINT4;\nDROP VARIABLE YINT4;\nDROP VARIABLE XINT8; \nDROP VARIABLE YINT8;\nCREATE VARIABLE XINT2 INT2 DEFAULT(1);\nCREATE VARIABLE YINT2 INT2 DEFAULT(3);\nCREATE VARIABLE XINT4 INT4 DEFAULT(1);\nCREATE VARIABLE YINT4 INT4 DEFAULT(3);\nCREATE VARIABLE XINT8 INT8 DEFAULT(1);\nCREATE VARIABLE YINT8 INT8 DEFAULT(3);')

get_ipython().run_cell_magic('sql', '', "WITH LOGIC(EXAMPLE, X, Y, RESULT) AS\n  (\n  VALUES\n     ('INT2AND(X,Y)',XINT2,YINT2,INT2AND(XINT2,YINT2)),\n     ('INT2OR(X,Y) ',XINT2,YINT2,INT2OR(XINT2,YINT2)),\n     ('INT2XOR(X,Y)',XINT2,YINT2,INT2XOR(XINT2,YINT2)),\n     ('INT2NOT(X)  ',XINT2,YINT2,INT2NOT(XINT2))\n  )\nSELECT * FROM LOGIC")

get_ipython().run_cell_magic('sql', '', "WITH LOGIC(EXAMPLE, X, Y, RESULT) AS\n  (\n  VALUES\n     ('INT4AND(X,Y)',XINT4,YINT4,INT4AND(XINT4,YINT4)),\n     ('INT4OR(X,Y) ',XINT4,YINT4,INT4OR(XINT4,YINT4)),\n     ('INT4XOR(X,Y)',XINT4,YINT4,INT4XOR(XINT4,YINT4)),\n     ('INT4NOT(X)  ',XINT4,YINT4,INT4NOT(XINT4))\n  )\nSELECT * FROM LOGIC")

get_ipython().run_cell_magic('sql', '', "WITH LOGIC(EXAMPLE, X, Y, RESULT) AS\n  (\n  VALUES\n     ('INT8AND(X,Y)',XINT8,YINT8,INT8AND(XINT8,YINT8)),\n     ('INT8OR(X,Y) ',XINT8,YINT8,INT8OR(XINT8,YINT8)),\n     ('INT8XOR(X,Y)',XINT8,YINT8,INT8XOR(XINT8,YINT8)),\n     ('INT8NOT(X)  ',XINT8,YINT8,INT8NOT(XINT8))\n  )\nSELECT * FROM LOGIC")

get_ipython().magic('sql VALUES TO_HEX(255)')

get_ipython().magic("sql VALUES RAWTOHEX('Hello')")

get_ipython().magic('sql VALUES TO_HEX(12336)')

get_ipython().magic("sql VALUES RAWTOHEX('00');")

get_ipython().magic('sql -q DROP TABLE AS_EMP')
get_ipython().magic('sql CREATE TABLE AS_EMP AS (SELECT EMPNO, SALARY+BONUS FROM EMPLOYEE) DEFINITION ONLY;')

get_ipython().magic('sql -q DROP TABLE AS_EMP')
get_ipython().magic('sql CREATE TABLE AS_EMP AS (SELECT EMPNO, SALARY+BONUS AS PAY FROM EMPLOYEE) DEFINITION ONLY;')

get_ipython().run_cell_magic('sql', '', "SELECT DISTINCT(NAME), COLTYPE, LENGTH FROM SYSIBM.SYSCOLUMNS \n  WHERE TBNAME='AS_EMP' AND TBCREATOR=CURRENT USER")

get_ipython().magic('sql -q DROP TABLE AS_EMP')
get_ipython().magic('sql CREATE TABLE AS_EMP AS (SELECT EMPNO, SALARY+BONUS AS PAY FROM EMPLOYEE) WITH DATA;')

get_ipython().run_cell_magic('sql', '-q', "DROP TABLE AS_EMP;\nCREATE TABLE AS_EMP(LAST,PAY) AS \n (\n SELECT LASTNAME, SALARY FROM EMPLOYEE \n    WHERE WORKDEPT='D11'\n FETCH FIRST 3 ROWS ONLY\n ) WITH DATA;")

get_ipython().run_cell_magic('sql', '-q', 'DROP TABLE AS_EMP;\nCREATE TABLE AS_EMP(DEPARTMENT, LASTNAME) AS \n  (SELECT WORKDEPT, LASTNAME FROM EMPLOYEE\n     OFFSET 5 ROWS\n     FETCH FIRST 10 ROWS ONLY\n  ) WITH DATA;\nSELECT * FROM AS_EMP;')

