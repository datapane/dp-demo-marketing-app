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


df_orders = pd.read_csv("data/order.csv").set_index("Name")
df_items = pd.read_csv("data/items.csv", low_memory=False).set_index("Name")
df_customers = pd.read_csv("data/cust.csv").set_index("Cust_ID")

with open("data/zipcode_lookup.json", "r") as f:
    zipcode_lookup = json.load(f)

df_zipcode_lookup = pd.DataFrame(zipcode_lookup).T

datetime_now = datetime.datetime(
    2023, 4, 27, 10, 15, 0, 0, tzinfo=ZoneInfo("US/Pacific")
)

a.set_timezones(df_orders, ["Created at"])
a.set_timezones(df_items, ["Created at"])
a.set_timezones(df_customers, ["first_order", "last_order"])

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def get_summary_stats(df_orders_window, df_customers_window, df_orders_window_previous, df_customers_window_previous):

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

    def gen_bn(title, column):
        return dp.BigNumber(
            title,
            f"{stats_current_period[column].item():n}",
            f"{stats_delta[column].item():n}",
            is_upward_change=stats_upward_change[column].item(),
        )
    
    block_summary_stats = dp.Group(
        gen_bn("Orders Created", 'orders'),
        gen_bn("Sales Completed", 'sales'),
        gen_bn("New Customers", 'new_customers'),
        gen_bn("Returning Customers", 'returning_customers'),
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

    return block_summary_stats

def gen_audience_plots(df_orders_window):

    orders_by_customer = a.orders_by_customer(df_orders_window)

    orders_by_day = a.orders_by_day(df_orders_window)

    audience_plots = dp.Group(
        dp.Plot(a.plot_value_counts(
            orders_by_customer,
            title=f"Total number of orders: {len(df_orders_window)}",
        )),
        dp.Plot(a.plot_value_counts(
            orders_by_day, title="Orders by day of week"
        )),
       # bug TODO dp.Plot(a.plot_aov_histogram(df_orders_window)),
        columns=3,
    )

    return audience_plots


def gen_top_product_plots(df_items_window, df_orders_window):
    
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

    top_city = (
        df_zipcode_lookup[df_zipcode_lookup.index.isin(zipcodes)]["place_name"]
        .value_counts()
        .index[0]
    )
    bn_top_city = dp.BigNumber("Top City", top_city)
    bn_top_city

    audience_tops = dp.Group(
        bn_top_product,
        bn_top_discount_code,
        bn_top_sku,
        bn_top_city,
    )

    return audience_tops


def render(start_date, end_date):

    window_end = pd.to_datetime(end_date).tz_localize('UTC')
    window_start = pd.to_datetime(start_date).tz_localize('UTC')

    df_orders_window, df_orders_window_previous = a.get_window(
        df_orders, "Created at", window_start, window_end
    )
    df_items_window, df_items_window_previous = a.get_window(
        df_items, "Created at", window_start, window_end
    )
    df_customers_window, df_customers_window_previous = a.get_window(
        df_customers, "first_order", window_start, window_end
    )
    
    summary_stats = get_summary_stats(df_orders_window, df_customers_window, df_orders_window_previous, df_customers_window_previous)
    #locations = a.plot_customer_locations(df_customers, 20, df_zipcode_lookup)
    
    top_10_products = (
        df_items_window["Lineitem name"]
        .value_counts()
        .rename_axis("unique_values")
        .to_frame("counts")
        .reset_index()
        .rename(columns={0: "counts"})
        .head(10)
    )
    
    frequent_combinations = a.frequent_product_combinations(df_items_window)
    
    retention_fig, df1, avg_order_fig, df2 = a.cohort_analysis(df_orders_window)
    
    v = dp.Blocks(
        audience_plots,
        dp.Select(
            dp.Group(
                audience_tops,
                label='Top products'
            ),
            dp.Group(
                dp.Table(frequent_combinations),
                dp.Plot(
                    a.plot_value_counts(top_10_products, "Top 10 Products")
                ),
                columns=2,
                widths=[6, 4],
                label='Market basket analysis'
            ),
            dp.Group(dp.Plot(retention_fig), df1, dp.Plot(avg_order_fig), df2, columns=2, label='Retention analysis'),
        )
    )
    
    return v

df_calmap = (
    (
        df_orders["Created at"]
        .dt.date.value_counts()
        .rename_axis("Date")
        .to_frame("Orders")
    )
    .reset_index()
    .rename(columns={0: "counts"})
)

df, year, last_sample_date = calendar_heatmap.wrangle_df(df_calmap, year=2023)
audience_plots = gen_audience_plots(df_orders)
audience_tops = gen_top_product_plots(df_items, df_orders)

root = dp.View(
    calendar_heatmap.plot_heatmap("Orders", df, legend=True, color_scheme="cividis"),
    audience_plots,
    audience_tops,
    dp.Form(
        on_submit=render,
        controls=dict(
            start_date=dp.Date("start", label="End date", initial=datetime.date.today()),
            end_date=dp.Date("end", label="Start date", initial=datetime.date.today() - relativedelta(months=1)),
        )
    )
)

dp.serve_app(root)
