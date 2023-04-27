import datapane as dp
from pathlib import Path

import pandas as pd
import duckdb

import analytics as a
import json
from datapane_components import section


@dp.task(name="update-db")
def update_db():
    # Load data via Singer / Meltano
    # load_data = subprocess.Popen("tap-shopify --config tap_config.json --catalog catalog.json | target-duckdb --config target_config.json >> state.json",
    #                              shell=True)

    # use local CSVs
    df_orders = pd.read_csv("data/order.csv").set_index("Name")
    df_items = pd.read_csv("data/items.csv", low_memory=False).set_index("Name")
    df_customers = pd.read_csv("data/cust.csv").set_index("Cust_ID")

    with open("data/zipcode_lookup.json", "r") as f:
        zipcode_lookup = json.load(f)
        df_zipcode_lookup = pd.DataFrame(zipcode_lookup).T
    # make our `datetime`s aware of the time zone.
    a.set_timezones(df_orders, ["Created at"])
    a.set_timezones(df_items, ["Created at"])
    a.set_timezones(df_customers, ["first_order", "last_order"])

    # load as tables into duck
    Path("data.db").unlink(missing_ok=True)
    duckdb.default_connection.execute("SET GLOBAL pandas_analyze_sample=100000")
    con = duckdb.connect("data.db")
    con.execute("CREATE TABLE orders AS SELECT * FROM df_orders")
    con.execute("CREATE TABLE items AS SELECT * FROM df_items")
    con.execute("CREATE TABLE customers AS SELECT * FROM df_customers")
    con.execute("CREATE TABLE zipcode_lookup AS SELECT * FROM df_zipcode_lookup")


@dp.task(name="daily-report")
def daily_report():
    report = dp.Group(
        "## Summary",
        a.gen_summary_stats(),
        *section("## Top Products"),
        a.gen_top_product_stats(),
        a.gen_audiencce_plots(),
        label="Top Stats",
    )

    dp.save_report(report, name="daily_report.html")

    # send report via slack and email
    dp.notification.slack(channel="#updates", file="daily_report.html")
    dp.notification.email(addresses=["leo@example.com", "mg@example.com"], file="daily_report.html")
