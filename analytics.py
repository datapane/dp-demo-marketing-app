import datetime as datetime
import locale
import typing as t
import warnings

import altair as alt
import folium
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dominate.tags import *
from folium import plugins
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

alt.data_transformers.enable("default", max_rows=None)
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
    fig = (
        alt.Chart(df)
        .mark_bar(color="#00B3FE")
        .encode(
            alt.X("Total:Q", bin=True, axis=alt.Axis(format="$f"), title=None),
            alt.Y("count()", title=None),
        )
        .properties(title=f"Average order value {locale.currency(df.Total.mean(), grouping=True)}")
    )
    return fig


def plot_value_counts(series: pd.DataFrame, title: str, scale: str = "linear", bar_color: str = "#5A5BC1") -> alt.Chart:
    fig = (
        alt.Chart(series)
        .mark_bar(color=bar_color)
        .encode(
            x=alt.X("unique_values:N", title=None, sort=None),
            y=alt.Y("counts:Q", scale=alt.Scale(type=scale), title=None),
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


def get_month(x: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(x.year, x.month, 1)


def cohort_analysis(
    df_orders_window: pd.DataFrame,
) -> t.Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    df_orders_cohort = df_orders_window

    df_orders_cohort["order_month"] = df_orders_cohort["Created at"].apply(get_month)

    grouping = df_orders_cohort.groupby("Cust_ID")["order_month"]

    df_orders_cohort["cohort_month"] = grouping.transform("min")

    def get_date_int(df: pd.DataFrame, column: str) -> t.Tuple[int, int, int]:
        year = df[column].dt.year
        month = df[column].dt.month
        day = df[column].dt.day
        return year, month, day

    # Getting the integers for date parts from the `InvoiceDay` column
    transcation_year, transaction_month, _ = get_date_int(df_orders_cohort, "order_month")

    # Getting the integers for date parts from the `CohortDay` column
    cohort_year, cohort_month, _ = get_date_int(df_orders_cohort, "cohort_month")

    #  Get the  difference in years
    years_diff = transcation_year - cohort_year

    # Calculate difference in months
    months_diff = transaction_month - cohort_month

    df_orders_cohort["cohort_index"] = years_diff * 12 + months_diff + 1

    # Counting daily active user from each chort
    grouping = df_orders_cohort.groupby(["cohort_month", "cohort_index"])

    # Counting number of unique customer Id's falling in each group of CohortMonth and CohortIndex
    cohort_data = grouping["Cust_ID"].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()

    # Assigning column names to the dataframe created above
    cohort_counts = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="Cust_ID")

    ### Retention rate
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    retention.index = retention.index.strftime("%Y-%m")

    retention_fig = plt.figure(figsize=(16, 10))
    plt.rc("font", size=20)
    plt.title(
        "Retention Rate in percentage: Monthly Cohorts",
    )
    sns.heatmap(
        retention,
        annot=True,
        fmt=".0%",
        cmap="cividis_r",
        vmin=0.0,
        vmax=0.6,
    )
    plt.ylabel(
        "Cohort Month",
    )
    plt.xlabel(
        "Cohort Index",
    )
    plt.yticks(rotation=360)
    plt.close()

    ### Average order total monthly cohort
    grouping = df_orders_cohort.groupby(["cohort_month", "cohort_index"])
    cohort_data = grouping["Total"].mean()
    cohort_data = cohort_data.reset_index()
    average_order = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="Total")

    average_standard_cost = average_order.round(1)
    average_standard_cost.index = average_standard_cost.index.strftime("%Y-%m")

    avg_order_fig = plt.figure(figsize=(16, 10))
    plt.title("Average Order Total: Monthly Cohorts")
    sns.heatmap(
        average_standard_cost,
        annot=True,
        vmin=0.0,
        vmax=60,
        cmap="cividis_r",
        fmt="g",
    )
    plt.ylabel("Cohort Month")
    plt.xlabel("Cohort Index")
    plt.yticks(rotation=360)
    plt.xticks(fontsize=20)
    plt.close()

    return retention_fig, avg_order_fig
