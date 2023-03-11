from pathlib import Path

import pandas as pd
import duckdb

import analytics as a


def import_into_db():
    df_orders = pd.read_csv("data/order.csv").set_index("Name")
    df_items = pd.read_csv("data/items.csv", low_memory=False).set_index("Name")
    df_customers = pd.read_csv("data/cust.csv").set_index("Cust_ID")

    #    with open("data/zipcode_lookup.json", "r") as f:
    #        zipcode_lookup = json.load(f)
    #    df_zipcode_lookup = pd.DataFrame(zipcode_lookup).T
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
    # con.execute("CREATE TABLE zipcode_lookup AS SELECT * from read_json_auto('data/zipcode_lookup.json', maximum_object_size=10485760)")
    # con.execute('CREATE TABLE zipcode_lookup AS SELECT * FROM df_zipcode_lookup')


if __name__ == "__main__":
    import_into_db()
