import datetime
import locale
import textwrap
import json

import datapane as dp
import numpy as np
import pandas as pd
from datapane_components import calendar_heatmap, section

import analytics as a

################################################################################
# Global Dataset
df_orders = pd.read_csv("data/order.csv.gz").set_index("Name")
df_items = pd.read_csv("data/items.csv.gz", low_memory=False).set_index("Name")
df_customers = pd.read_csv("data/cust.csv.gz").set_index("Cust_ID")
with open("data/zipcode_lookup.json", "r") as f:
    df_zipcode_lookup = pd.DataFrame(json.load(f)).T

a.set_timezones(df_orders, ["Created at"])
a.set_timezones(df_items, ["Created at"])
a.set_timezones(df_customers, ["first_order", "last_order"])


################################################################################
# Summary stats
# 30 day stats (sales, aov, new customers, new orders, etc.)
def gen_summary_stats(
    df_orders_window: pd.DataFrame,
    df_customers_window: pd.DataFrame,
    df_orders_window_previous: pd.DataFrame,
    df_customers_window_previous: pd.DataFrame,
) -> dp.Group:
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
    return block_summary_stats


################################################################################
# Audiences
# Top 10% of customers, Most frequent purchasers, top country, top product, etc.
def gen_audiencce_plots(df_orders_window: pd.DataFrame) -> dp.Group:
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
                bar_color="#5A5BC1",
                scale="log",
            )
        ),
        dp.Plot(a.plot_value_counts(orders_by_day, title="Orders by day of week", bar_color="#E7088E")),
        dp.Plot(a.plot_aov_histogram(df_orders_window)),
        columns=3,
    )

    return audience_plots


################################################################################
# Top Product (Big Numbers)
def gen_top_product_stats(
    df_items_window: pd.DataFrame, df_orders_window: pd.DataFrame, df_customers_window: pd.DataFrame
) -> dp.Group:
    plot_customer_locations = a.plot_customer_locations(df_customers_window, 20, df_zipcode_lookup)

    top_product = textwrap.shorten(
        df_items_window["Lineitem name"].value_counts().index[0],
        width=20,
        placeholder="...",
    )
    bn_top_product = dp.BigNumber("Top Product", top_product)

    # Top SKU
    top_sku = df_items_window["Lineitem sku"].value_counts().index[2]
    bn_top_sku = dp.BigNumber("Top SKU", top_sku)

    # Top Discount Code
    top_discount_code = df_items_window["Discount Code"].value_counts().index[2]
    bn_top_discount_code = dp.BigNumber("Top Discount Code", top_discount_code)

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

    audience_tops = dp.Group(
        bn_top_product,
        bn_top_discount_code,
        bn_top_sku,
        bn_top_city,
    )

    return dp.Group(plot_customer_locations, dp.Empty(name="zyz"), audience_tops, columns=2, widths=[2, 1])


################################################################################
# Market Basket
# Frequency of popular items
def gen_popular_items(df_items_window: pd.DataFrame) -> dp.Group:
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

    popular = dp.Group(
        dp.Table(frequent_combinations),
        dp.Plot(a.plot_value_counts(top_10_products, "Top 10 Products", bar_color="#4340B1")),
        columns=2,
        widths=[6, 4],
    )
    return popular


################################################################################
# Cohort analysis
def gen_cohort_analysis(df_orders_window: pd.DataFrame) -> dp.Group:
    df_calmap = (
        (df_orders_window["Created at"].dt.date.value_counts().rename_axis("Date").to_frame("Orders"))
        .reset_index()
        .rename(columns={0: "counts"})
    )

    df, year, last_sample_date = calendar_heatmap.wrangle_df(df_calmap, year=2023)
    cal_plot = calendar_heatmap.plot_heatmap("Orders", df, legend=True, color_scheme="cividis")

    retention_fig, avg_order_fig = a.cohort_analysis(df_orders_window)

    return dp.Group(cal_plot, dp.Group(dp.Plot(retention_fig), dp.Plot(avg_order_fig), columns=2))


################################################################################
# DP App
def render(start_date: datetime.date, end_date: datetime.date, all_data: bool) -> dp.View:
    # get the data window
    if all_data:
        window_start = df_orders["Created at"].min() + datetime.timedelta(weeks=1)
        window_end = df_orders["Created at"].max()
    else:
        window_start = pd.to_datetime(start_date).tz_localize("US/Pacific")
        window_end = pd.to_datetime(end_date).tz_localize("US/Pacific")

    df_orders_window, df_orders_window_previous = a.get_window(df_orders, "Created at", window_start, window_end)
    df_items_window, df_items_window_previous = a.get_window(df_items, "Created at", window_start, window_end)
    df_customers_window, df_customers_window_previous = a.get_window(
        df_customers, "first_order", window_start, window_end
    )

    tab1 = dp.Group(
        "## Summary",
        gen_summary_stats(
            df_orders_window, df_customers_window, df_orders_window_previous, df_customers_window_previous
        ),
        *section("## Top Products"),
        gen_top_product_stats(df_items_window, df_orders_window, df_customers_window),
        gen_audiencce_plots(df_orders_window),
        label="Top Stats",
    )

    tab2 = dp.Group(gen_popular_items(df_items_window), label="Popular Items")

    tab3 = dp.Group(gen_cohort_analysis(df_orders_window), label="Cohort Analysis")

    tab4 = dp.Group(f"## Sales data for {start_date} to {end_date}", dp.DataTable(df_orders_window), label="Order Data")

    dp.save_report(tab1, "sales_report.html")

    return dp.View(
        dp.Toggle(dp.Attachment(file="sales_report.html"), name="download", label="Download standalone Report"),
        dp.Select(tab1, tab2, tab3, tab4, name="main_results"),
    )


initial_view = dp.View(
    "# Marketing App",
    dp.Media("./logo.jpg"),
    dp.Form(
        on_submit=render,
        label="Select the order range to run analysis over",
        controls=dp.Controls(
            label="Enter a date range OR select the full dataset",
            start_date=dp.Date(
                "start", label="Start date", initial=datetime.date.today() - datetime.timedelta(weeks=26)
            ),
            end_date=dp.Date("end", label="End date", initial=datetime.date.today()),
            all_data=dp.Switch("All Data", label="Use the full dataset", initial=False),
        ),
    ),
)

dp.enable_logging()
dp.serve_app(initial_view)
