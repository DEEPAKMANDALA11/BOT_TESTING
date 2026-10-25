-- DUMMY TEST DATA ONLY
-- Creates a landing table used as the source for stg.tableA.
CREATE DATABASE IF NOT EXISTS landing;

CREATE TABLE IF NOT EXISTS landing.tableA_raw (
    customer_id       INTEGER,
    customer_name     VARCHAR(100),
    status_code       VARCHAR(20),
    transaction_amount DECIMAL(12,2),
    load_date         DATE
);
