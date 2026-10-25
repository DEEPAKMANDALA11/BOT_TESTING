-- DUMMY TEST DATA ONLY
-- Main load: landing.tableA_raw -> stg.tableA
CREATE DATABASE IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.tableA (
    customer_id        INTEGER,
    customer_name      VARCHAR(100),
    customer_status    VARCHAR(30),
    transaction_amount DECIMAL(12,2),
    load_date          DATE,
    load_source        VARCHAR(30)
);

INSERT INTO stg.tableA (
    customer_id,
    customer_name,
    customer_status,
    transaction_amount,
    load_date,
    load_source
)
SELECT
    src.customer_id,
    UPPER(TRIM(src.customer_name)) AS customer_name,
    CASE
        WHEN src.status_code = 'A' THEN 'ACTIVE'
        WHEN src.status_code = 'I' THEN 'INACTIVE'
        ELSE 'UNKNOWN'
    END AS customer_status,
    COALESCE(src.transaction_amount, 0.00) AS transaction_amount,
    src.load_date,
    'TABLEA_SQL_TEST' AS load_source
FROM landing.tableA_raw AS src
WHERE src.customer_id IS NOT NULL;
