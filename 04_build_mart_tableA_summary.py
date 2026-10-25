"""DUMMY TEST DATA ONLY.

Downstream PySpark example: core.tableA -> mart.tableA_summary.
This file is illustrative and requires a Spark environment to execute.
"""

from pyspark.sql import functions as F


def build_tableA_summary(spark):
    source_df = spark.read.table("core.tableA")

    summary_df = (
        source_df
        .groupBy("customer_status")
        .agg(
            F.count("*").alias("customer_count"),
            F.sum("transaction_amount").alias("total_transaction_amount"),
        )
        .withColumn("summary_run_type", F.lit("TABLEA_PYTHON_TEST"))
    )

    (
        summary_df.write
        .mode("overwrite")
        .saveAsTable("mart.tableA_summary")
    )

    return summary_df
