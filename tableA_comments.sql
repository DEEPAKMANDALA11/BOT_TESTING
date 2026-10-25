-- DUMMY TEST DATA ONLY
-- Test data dictionary / column comments for stg.tableA.
COMMENT ON TABLE stg.tableA IS 'Dummy staging table used to test the repository lineage chatbot';

COMMENT ON COLUMN stg.tableA.customer_id IS 'Synthetic customer identifier; required for the test load';
COMMENT ON COLUMN stg.tableA.customer_name IS 'Customer name trimmed and converted to uppercase';
COMMENT ON COLUMN stg.tableA.customer_status IS 'Derived from landing status_code: A=ACTIVE, I=INACTIVE, otherwise UNKNOWN';
COMMENT ON COLUMN stg.tableA.transaction_amount IS 'Synthetic transaction amount; NULL is replaced with zero';
COMMENT ON COLUMN stg.tableA.load_date IS 'Load date inherited from landing.tableA_raw';
COMMENT ON COLUMN stg.tableA.load_source IS 'Literal audit value identifying the SQL test load';
