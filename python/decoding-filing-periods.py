from calaccess_processed.models.tracking import ProcessedDataVersion

ProcessedDataVersion.objects.latest()

from project import sql_to_agate

sql_to_agate(
    """
    SELECT UPPER("STMT_TYPE"), COUNT(*)
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD"
    WHERE "FORM_TYPE" = 'F460'
    GROUP BY 1
    ORDER BY COUNT(*) DESC;
    """
).print_table()

sql_to_agate(
    """
    SELECT FF."STMNT_TYPE", LU."CODE_DESC", COUNT(*)
    FROM "FILER_FILINGS_CD" FF
    JOIN "LOOKUP_CODES_CD" LU
    ON FF."STMNT_TYPE" = LU."CODE_ID"
    AND LU."CODE_TYPE" = 10000
    GROUP BY 1, 2;
    """
).print_table()

sql_to_agate(
    """
    SELECT *
    FROM "FILING_PERIOD_CD"
    """
).print_table()

sql_to_agate(
    """
    SELECT "PERIOD_DESC", COUNT(*)
    FROM "FILING_PERIOD_CD"
    GROUP BY 1;
    """
).print_table()

sql_to_agate(
    """
    SELECT "END_DATE" - "START_DATE" AS duration, COUNT(*)
    FROM "FILING_PERIOD_CD"
    GROUP BY 1;
    """
).print_table()

sql_to_agate(
    """
    SELECT DATE_PART('year', "START_DATE")::int as year, COUNT(*)
    FROM "FILING_PERIOD_CD"
    GROUP BY 1
    ORDER BY 1 DESC;
    """
).print_table()

sql_to_agate(
    """
    SELECT ff."PERIOD_ID", fp."START_DATE", fp."END_DATE", fp."PERIOD_DESC", COUNT(*)
    FROM "FILER_FILINGS_CD" ff
    JOIN "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    ON ff."FILING_ID" = cvr."FILING_ID"
    AND ff."FILING_SEQUENCE" = cvr."AMEND_ID"
    AND cvr."FORM_TYPE" = 'F460'
    JOIN "FILING_PERIOD_CD" fp
    ON ff."PERIOD_ID" = fp."PERIOD_ID"
    GROUP BY 1, 2, 3, 4
    ORDER BY fp."START_DATE" DESC;
    """
).print_table()

sql_to_agate(
    """
    SELECT cvr."FILING_ID", cvr."FORM_TYPE", cvr."FILER_NAML"
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    LEFT JOIN  "FILER_FILINGS_CD" ff
    ON cvr."FILING_ID" = ff."FILING_ID"
    AND cvr."AMEND_ID" = ff."FILING_SEQUENCE" 
    WHERE cvr."FORM_TYPE" = 'F460'
    AND (ff."FILING_ID" IS NULL OR ff."FILING_SEQUENCE" IS NULL)
    ORDER BY cvr."FILING_ID";
    """
).print_table(max_column_width=60)

sql_to_agate(
    """
    SELECT 
        CASE 
            WHEN cvr."FROM_DATE" < fp."START_DATE" THEN 'filing from_date before period start_date'
            WHEN cvr."THRU_DATE" > fp."END_DATE" THEN 'filing thru_date after period end_date'
            ELSE 'okay'
        END as test,
        COUNT(*) 
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    JOIN "FILER_FILINGS_CD" ff
    ON cvr."FILING_ID" = ff."FILING_ID"
    AND cvr."AMEND_ID" = ff."FILING_SEQUENCE"
    JOIN "FILING_PERIOD_CD" fp
    ON ff."PERIOD_ID" = fp."PERIOD_ID"
    WHERE cvr."FORM_TYPE" = 'F460'
    GROUP BY 1;
    """
).print_table(max_column_width=60)

sql_to_agate(
    """
    SELECT 
            cvr."THRU_DATE" - fp."END_DATE" as date_diff,
            COUNT(*) 
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    JOIN "FILER_FILINGS_CD" ff
    ON cvr."FILING_ID" = ff."FILING_ID"
    AND cvr."AMEND_ID" = ff."FILING_SEQUENCE"
    JOIN "FILING_PERIOD_CD" fp
    ON ff."PERIOD_ID" = fp."PERIOD_ID"
    WHERE cvr."FORM_TYPE" = 'F460'
    AND cvr."THRU_DATE" > fp."END_DATE"
    GROUP BY 1
    ORDER BY COUNT(*) DESC;
    """
).print_table(max_column_width=60)

sql_to_agate(
    """
    SELECT 
            cvr."FILING_ID",
            cvr."AMEND_ID",
            cvr."FROM_DATE",
            cvr."THRU_DATE",
            fp."START_DATE",
            fp."END_DATE"
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    JOIN "FILER_FILINGS_CD" ff
    ON cvr."FILING_ID" = ff."FILING_ID"
    AND cvr."AMEND_ID" = ff."FILING_SEQUENCE"
    JOIN "FILING_PERIOD_CD" fp
    ON ff."PERIOD_ID" = fp."PERIOD_ID"
    WHERE cvr."FORM_TYPE" = 'F460'
    AND 90 < cvr."THRU_DATE" - fp."END_DATE" 
    AND cvr."THRU_DATE" - fp."END_DATE" < 93
    ORDER BY cvr."THRU_DATE" DESC;
    """
).print_table(max_column_width=60)

sql_to_agate(
    """
    SELECT UPPER(cvr."STMT_TYPE"), COUNT(*)
    FROM "CVR_CAMPAIGN_DISCLOSURE_CD" cvr
    JOIN "FILER_FILINGS_CD" ff
    ON cvr."FILING_ID" = ff."FILING_ID"
    AND cvr."AMEND_ID" = ff."FILING_SEQUENCE"
    JOIN "FILING_PERIOD_CD" fp
    ON ff."PERIOD_ID" = fp."PERIOD_ID"
    WHERE cvr."FORM_TYPE" = 'F460'
    AND 90 < cvr."THRU_DATE" - fp."END_DATE" 
    AND cvr."THRU_DATE" - fp."END_DATE" < 93
    GROUP BY 1
    ORDER BY COUNT(*) DESC;
    """
).print_table(max_column_width=60)



