import datetime as datetime
import locale
import typing as t
import warnings

import altair as alt
import folium
import pandas as pd
from dominate.tags import *
from folium import plugins
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def set_timezones(df: pd.DataFrame, cols: t.List[str]) -> None:
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df[col].dt.tz_convert("US/Pacific")


def get_window(
    df: pd.DataFrame,
    date_col: str,
    window_start: datetime.datetime,
    window_end: datetime.datetime,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    window = df[(df[date_col] > window_start) & (df[date_col] < window_end)]
    previous_period_window = df[
        (df[date_col] > window_start - (window_end - window_start)) & (df[date_col] < window_start)
    ]
    return window, previous_period_window


def summary_stats(df_orders: pd.DataFrame, df_customers: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    stats["orders"] = len(df_orders)
    stats["sales"] = len(df_orders[df_orders["Financial Status"] == "paid"])
    stats["aov"] = df_orders.Total.mean()
    stats["revenue"] = df_orders.Total.sum()
    stats["new_customers"] = len(df_customers)
    stats["returning_customers"] = len(df_orders.Cust_ID.unique()) - stats["new_customers"]
    return pd.DataFrame.from_dict(stats, orient="index").T


def get_summary_stats(
    df_orders_window: pd.DataFrame,
    df_customers_window: pd.DataFrame,
    df_orders_window_previous: pd.DataFrame,
    df_customers_window_previous: pd.DataFrame,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_current_period = summary_stats(df_orders_window, df_customers_window)
    stats_previous_period = summary_stats(df_orders_window_previous, df_customers_window_previous)
    stats_delta = stats_current_period - stats_previous_period
    stats_upward_change = stats_delta > 0

    return stats_current_period, stats_previous_period, stats_delta, stats_upward_change


def plot_customer_locations(
    df_customers: pd.DataFrame, order_threshold: int, df_zipcode_lookup: pd.DataFrame
) -> folium.Map:
    places = df_zipcode_lookup["state_name"].value_counts()
    places_mask = places > order_threshold
    popular_places = places[places_mask]
    df_popular_zipcode_lookup = df_zipcode_lookup.loc[df_zipcode_lookup["state_name"].isin(popular_places.index)]

    m = folium.Map(
        location=[
            df_popular_zipcode_lookup["latitude"].mean(),
            df_popular_zipcode_lookup["longitude"].mean(),
        ],
        tiles="OpenStreetMap",
        zoom_start=4,
    )

    locations = list(zip(df_popular_zipcode_lookup.latitude, df_popular_zipcode_lookup.longitude))
    cluster = plugins.MarkerCluster(locations=locations, popups=df_popular_zipcode_lookup["place_name"].tolist())

    m.add_child(cluster)
    return m


def plot_aov_histogram(df: pd.DataFrame) -> alt.Chart:
    # Calculate the range of the data and create 10 equal-width bins
    data_range = df["Total"].max() - df["Total"].min()
    bin_width = data_range / 10
    bins = [df["Total"].min() + i * bin_width for i in range(11)]

    # Bin the data using pd.cut() with the calculated bins
    binned_data = pd.cut(df["Total"], bins)

    # Group by the bins and count the number of items in each bin
    bin_counts = df.groupby(binned_data).size().reset_index(name="count")

    # Update the Altair chart to use the new bin_counts DataFrame
    fig = (
        alt.Chart(bin_counts)
        .mark_bar()
        .encode(
            alt.X("Total:O", title=None),
            alt.Y("count:Q", title=None),
        )
        .properties(title=f"Average order value {locale.currency(df.Total.mean(), grouping=True)}")
    )
    return fig


def plot_value_counts(series: pd.DataFrame, title: str) -> alt.Chart:
    fig = (
        alt.Chart(series)
        .mark_bar()
        .encode(
            x=alt.X("unique_values:N", title=None, sort=None),
            y=alt.Y("counts:Q", title=None),
        )
        .properties(title=title)
    )

    return fig


def to_unordered_list(items: t.List[str]) -> str:
    unordered_list = ul(style="margin:0;padding-left:20px")

    for item in items:
        unordered_list += li(item)

    return unordered_list.render(pretty=False)


def frequent_product_combinations(df_items_window: pd.DataFrame) -> pd.DataFrame:
    one_hot_encoded = (pd.get_dummies(df_items_window["Lineitem name"]).groupby("Name").sum()).clip(upper=1)

    # filter for only orders with 2 or more items
    one_hot_encoded_filtered = one_hot_encoded[one_hot_encoded.sum(axis=1) >= 2]

    frequent_itemsets = apriori(one_hot_encoded_filtered, min_support=0.025, use_colnames=True).sort_values(
        "support", ascending=False
    )

    assoc_rules = (
        association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
    )

    assoc_rules["Top Product Combinations"] = assoc_rules.apply(
        lambda row: to_unordered_list(sorted(list(row["antecedents"]) + list(row["consequents"]))),
        axis=1,
    )

    frequent_combinations = (
        assoc_rules[["Top Product Combinations", "support"]]
        .drop_duplicates("Top Product Combinations", keep="first")
        .sort_values("support", ascending=False)
    )

    frequent_combinations["% of orders"] = round(frequent_combinations["support"] * 100, 2)

    frequent_combinations = frequent_combinations.drop(["support"], axis=1).reset_index(drop=True).head(5)

    frequent_combinations.index = frequent_combinations.index + 1

    frequent_combinations = frequent_combinations.style.set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
            "vertical-align": "middle",
        }
    )

    frequent_combinations.format(na_rep="MISS", precision=2)

    row_hover = {"selector": "tr:hover", "props": [("background-color", "#9AE8FF")]}
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #4340B1; color: white;",
    }
    frequent_combinations.set_table_styles([row_hover, index_names, headers])

    return frequent_combinations


def orders_by_customer(df):
    orders = (
        (df["Cust_ID"].value_counts().value_counts().rename_axis("unique_values").to_frame("counts"))
        .reset_index()
        .rename(columns={0: "counts"})
    )
    return orders


def orders_by_day(df):
    orders = (
        (df["Created at"].dt.day_name().value_counts().rename_axis("unique_values").to_frame("counts"))
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .reset_index()
        .rename(columns={0: "counts"})
    )
    return orders


def get_month(x: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(x.year, x.month, 1)


def cohort_analysis(
    df_orders_window: pd.DataFrame,
) -> t.Tuple[alt.Chart, alt.Chart]:
    df_orders_cohort = df_orders_window

    df_orders_cohort["order_month"] = df_orders_cohort["Created at"].apply(get_month)

    grouping = df_orders_cohort.groupby("Cust_ID")["order_month"]

    df_orders_cohort["cohort_month"] = grouping.transform("min")

    def get_date_int(df: pd.DataFrame, column: str) -> t.Tuple[int, int, int]:
        year = df[column].dt.year
        month = df[column].dt.month
        day = df[column].dt.day
        return year, month, day

    transcation_year, transaction_month, _ = get_date_int(df_orders_cohort, "order_month")

    cohort_year, cohort_month, _ = get_date_int(df_orders_cohort, "cohort_month")

    years_diff = transcation_year - cohort_year

    months_diff = transaction_month - cohort_month

    df_orders_cohort["cohort_index"] = years_diff * 12 + months_diff + 1

    grouping = df_orders_cohort.groupby(["cohort_month", "cohort_index"])

    cohort_data = grouping["Cust_ID"].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()

    cohort_counts = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="Cust_ID")

    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    retention.index = retention.index.strftime("%Y-%m")

    retention = retention.reset_index().melt("cohort_month", var_name="cohort_index", value_name="retention_rate")

    retention_chart = (
        alt.Chart(retention)
        .mark_rect()
        .encode(
            x=alt.X("cohort_index:O", title="Cohort Index"),
            y=alt.Y("cohort_month:O", title="Cohort Month"),
            color=alt.Color("retention_rate:Q", legend=alt.Legend(format=".0%")),
            tooltip=[
                alt.Tooltip("cohort_month:O", title="Cohort Month"),
                alt.Tooltip("cohort_index:O", title="Cohort Index"),
                alt.Tooltip("retention_rate:Q", format=".0%", title="Retention Rate"),
            ],
        )
        .properties(width=600, height=400, title="Retention Rate in percentage: Monthly Cohorts")
    )

    grouping = df_orders_cohort.groupby(["cohort_month", "cohort_index"])
    cohort_data = grouping["Total"].mean()
    cohort_data = cohort_data.reset_index()
    average_order = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="Total")

    average_standard_cost = average_order.round(1)
    average_standard_cost.index = average_standard_cost.index.strftime("%Y-%m")

    average_standard_cost = average_standard_cost.reset_index().melt(
        "cohort_month", var_name="cohort_index", value_name="average_order_total"
    )

    avg_order_chart = (
        alt.Chart(average_standard_cost)
        .mark_rect()
        .encode(
            x=alt.X("cohort_index:O", title="Cohort Index"),
            y=alt.Y("cohort_month:O", title="Cohort Month"),
            color=alt.Color("average_order_total:Q", legend=alt.Legend(format="g")),
            tooltip=[
                alt.Tooltip("cohort_month:O", title="Cohort Month"),
                alt.Tooltip("cohort_index:O", title="Cohort Index"),
                alt.Tooltip("average_order_total:Q", format=".1f", title="Average Order Total"),
            ],
        )
        .properties(width=600, height=400, title="Average Order Total: Monthly Cohorts")
    )

    return retention_chart, cohort_data, avg_order_chart, average_standard_cost
