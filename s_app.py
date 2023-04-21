import datetime
import json
import locale
import os
import textwrap
from zoneinfo import ZoneInfo

import datapane as dp
import numpy as np
import pandas as pd
from datapane_components import calendar_heatmap
from dateutil.relativedelta import relativedelta

import analytics as a


# ## Dataset
df_orders = pd.read_csv("data/order.csv").set_index("Name")
df_items = pd.read_csv("data/items.csv", low_memory=False).set_index("Name")
df_customers = pd.read_csv("data/cust.csv").set_index("Cust_ID")

with open("data/zipcode_lookup.json", "r") as f:
    zipcode_lookup = json.load(f)

df_zipcode_lookup = pd.DataFrame(zipcode_lookup).T

# Dates
# We'll create `datetime_now` to simulate running `datetime.now()` at the start of the PyData slot.

datetime_now = datetime.datetime(2023, 4, 27, 10, 15, 0, 0, tzinfo=ZoneInfo("US/Pacific"))
datetime_now


# And we'll make our `datetime`s aware of the time zone.
a.set_timezones(df_orders, ["Created at"])
a.set_timezones(df_items, ["Created at"])
a.set_timezones(df_customers, ["first_order", "last_order"])
# ## Currency
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


# ## Window

# Define a 30-day window.
window_end = datetime_now
window_start = datetime_now - relativedelta(months=3)

df_orders_window, df_orders_window_previous = a.get_window(df_orders, "Created at", window_start, window_end)
df_items_window, df_items_window_previous = a.get_window(df_items, "Created at", window_start, window_end)
df_customers_window, df_customers_window_previous = a.get_window(df_customers, "first_order", window_start, window_end)


# ## Summary stats
# 30 day stats (sales, aov, new customers, new orders, etc.)
(
    stats_current_period,
    stats_previous_period,
    stats_delta,
    stats_upward_change,
) = a.get_summary_stats(
    df_orders_window,
    df_customers_window,
    df_orders_window_previous,
    df_customers_window_previous,
)

block_summary_stats = dp.Group(
    dp.BigNumber(
        "Orders Created",
        f"{stats_current_period['orders'].item():n}",
        f"{stats_delta['orders'].item():n}",
        is_upward_change=stats_upward_change["orders"].item(),
    ),
    dp.BigNumber(
        "Sales Completed",
        f"{stats_current_period['sales'].item():n}",
        f"{stats_delta['sales'].item():n}",
        is_upward_change=stats_upward_change["sales"].item(),
    ),
    dp.BigNumber(
        "New Customers",
        f"{stats_current_period['new_customers'].item():n}",
        f"{stats_delta['new_customers'].item():n}",
        is_upward_change=stats_upward_change["new_customers"].item(),
    ),
    dp.BigNumber(
        "Returning Customers",
        f"{stats_current_period['returning_customers'].item():n}",
        f"{stats_delta['returning_customers'].item():n}",
        is_upward_change=stats_upward_change["returning_customers"].item(),
    ),
    dp.BigNumber(
        "Revenue Generated",
        locale.currency(stats_current_period["revenue"].item(), grouping=True),
        locale.currency(stats_delta["revenue"].item(), grouping=True),
        is_upward_change=stats_upward_change["revenue"].item(),
    ),
    dp.BigNumber(
        "AOV",
        locale.currency(stats_current_period["aov"].item(), grouping=True),
        locale.currency(stats_delta["aov"].item(), grouping=True),
        is_upward_change=stats_upward_change["aov"].item(),
    ),
    columns=3,
)

# block_summary_stats


# Audiences
# Top 10% of customers, Most frequent purchasers, top country, top product, etc.

plot_customer_locations = a.plot_customer_locations(df_customers, 20, df_zipcode_lookup)
# plot_customer_locations


orders_by_customer = (
    (df_orders_window["Cust_ID"].value_counts().value_counts().rename_axis("unique_values").to_frame("counts"))
    .reset_index()
    .rename(columns={0: "counts"})
)

orders_by_day = (
    (df_orders_window["Created at"].dt.day_name().value_counts().rename_axis("unique_values").to_frame("counts"))
    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    .reset_index()
    .rename(columns={0: "counts"})
)

audience_plots = dp.Group(
    dp.Plot(
        a.plot_value_counts(
            orders_by_customer,
            title=f"Total number of orders: {len(df_orders_window)}",
            scale="log",
        )
    ),
    dp.Plot(a.plot_value_counts(orders_by_day, title="Orders by day of week", bar_color="#E7088E")),
    dp.Plot(a.plot_aov_histogram(df_orders_window)),
    columns=3,
)

# audience_plots


# Top Product
top_product = textwrap.shorten(
    df_items_window["Lineitem name"].value_counts().index[0],
    width=20,
    placeholder="...",
)
bn_top_product = dp.BigNumber("Top Product", top_product)
bn_top_product

# Top SKU
top_sku = df_items_window["Lineitem sku"].value_counts().index[2]
bn_top_sku = dp.BigNumber("Top SKU", top_sku)
bn_top_sku

# Top Discount Code
top_discount_code = df_items_window["Discount Code"].value_counts().index[2]
bn_top_discount_code = dp.BigNumber("Top Discount Code", top_discount_code)
bn_top_discount_code

# Top City
zipcodes = list(
    np.where(
        df_orders_window["Shipping Zip"].str.len() == 5,
        df_orders_window["Shipping Zip"],
        df_orders_window["Shipping Zip"].str[:5],
    )
)

top_city = df_zipcode_lookup[df_zipcode_lookup.index.isin(zipcodes)]["place_name"].value_counts().index[0]
bn_top_city = dp.BigNumber("Top City", top_city)
# bn_top_city

audience_tops = dp.Group(
    bn_top_product,
    bn_top_discount_code,
    bn_top_sku,
    bn_top_city,
)

# audience_tops

# todo - what is this?
dp.Group(
    dp.Group(plot_customer_locations, audience_tops, columns=2, widths=[2, 1]),
    audience_plots,
    columns=1,
)


# Market Basket
# Frequency of popular items

top_10_products = (
    df_items_window["Lineitem name"]
    .value_counts()
    .rename_axis("unique_values")
    .to_frame("counts")
    .reset_index()
    .rename(columns={0: "counts"})
    .head(10)
)


# Item combinations per order. Start with one-hot encoding.
frequent_combinations = a.frequent_product_combinations(df_items_window)

dp.Group(
    dp.Table(frequent_combinations),
    dp.Plot(a.plot_value_counts(top_10_products, "Top 10 Products", bar_color="#4340B1")),
    columns=2,
    widths=[6, 4],
)

df_calmap = (
    (df_orders_window["Created at"].dt.date.value_counts().rename_axis("Date").to_frame("Orders"))
    .reset_index()
    .rename(columns={0: "counts"})
)


df, year, last_sample_date = calendar_heatmap.wrangle_df(df_calmap, year=2023)
calendar_heatmap.plot_heatmap("Orders", df, legend=True, color_scheme="cividis")


# Cohort analysis

retention_fig, avg_order_fig = a.cohort_analysis(df_orders_window)
dp.Group(dp.Plot(retention_fig), dp.Plot(avg_order_fig), columns=2)


def f(Window: str):
    if Window == "Last week":
        delta = relativedelta(weeks=1)
    elif Window == "Last month":
        delta = relativedelta(months=1)
    elif Window == "Last 3 months":
        delta = relativedelta(months=3)
    elif Window == "Last year":
        delta = relativedelta(year=1)

    return dp.Text(str(delta))


form = dp.Form(
    on_submit=f,
    controls=dict(
        Window=dp.Choice(
            "choice",
            initial="Last month",
            options=["Last week", "Last month", "Last 3 months", "Last year"],
        )
    ),
)


v = dp.Blocks(
    form,
    dp.Group(
        dp.Group(plot_customer_locations, audience_tops, columns=2, widths=[2, 1]),
        audience_plots,
        columns=1,
    ),
    dp.Group(
        dp.Table(frequent_combinations),
        dp.Plot(a.plot_value_counts(top_10_products, "Top 10 Products", bar_color="#4340B1")),
        columns=2,
        widths=[6, 4],
    ),
    dp.Group(calendar_heatmap.plot_heatmap("Orders", df, legend=True, color_scheme="cividis")),
    dp.Group(dp.Plot(retention_fig), dp.Plot(avg_order_fig), columns=2),
)

# v

dp.serve_app(v)
