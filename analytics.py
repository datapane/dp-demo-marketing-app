import pandas as pd
import datapane as dp
import folium
import locale
from folium import plugins
import altair as alt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import calendar_heatmap # switch to calendar_heatmap component


import warnings

warnings.filterwarnings("ignore")

alt.data_transformers.enable("default", max_rows=None)
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def set_timezones(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df[col].dt.tz_convert("US/Pacific")


def get_window(df, date_col, window_start, window_end):
    window = df[(df[date_col] > window_start) & (df[date_col] < window_end)]
    previous_period_window = df[
        (df[date_col] > window_start - (window_end - window_start))
        & (df[date_col] < window_start)
    ]
    return window, previous_period_window


def summary_stats(df_orders, df_customers):
    stats = {}
    stats["orders"] = len(df_orders)
    stats["sales"] = len(df_orders[df_orders["Financial Status"] == "paid"])
    stats["aov"] = df_orders.Total.mean()
    stats["revenue"] = df_orders.Total.sum()
    stats["new_customers"] = len(df_customers)
    stats["returning_customers"] = (
        len(df_orders.Cust_ID.unique()) - stats["new_customers"]
    )
    return pd.DataFrame.from_dict(stats, orient="index").T


def get_summary_stats(
    df_orders_window,
    df_customers_window,
    df_orders_window_previous,
    df_customers_window_previous,
):
    stats_current_period = summary_stats(df_orders_window, df_customers_window)
    stats_previous_period = summary_stats(
        df_orders_window_previous, df_customers_window_previous
    )
    stats_delta = stats_current_period - stats_previous_period
    stats_upward_change = stats_delta > 0

    return stats_current_period, stats_previous_period, stats_delta, stats_upward_change


def plot_customer_locations(df_customers, order_threshold, df_zipcode_lookup):
    places = df_zipcode_lookup["state_name"].value_counts()
    places_mask = places > order_threshold
    popular_places = places[places_mask]
    df_popular_zipcode_lookup = df_zipcode_lookup.loc[
        df_zipcode_lookup["state_name"].isin(popular_places.index)
    ]

    m = folium.Map(
        location=[
            df_popular_zipcode_lookup["latitude"].mean(),
            df_popular_zipcode_lookup["longitude"].mean(),
        ],
        tiles="OpenStreetMap",
        zoom_start=4,
    )

    locations = list(
        zip(df_popular_zipcode_lookup.latitude, df_popular_zipcode_lookup.longitude)
    )
    cluster = plugins.MarkerCluster(
        locations=locations, popups=df_popular_zipcode_lookup["place_name"].tolist()
    )

    m.add_child(cluster)
    return m


def plot_aov_histogram(df):
    fig = (
        alt.Chart(df)
        .mark_bar(color="#00B3FE")
        .encode(
            alt.X("Total:Q", bin=True, axis=alt.Axis(format="$f"), title=None),
            alt.Y("count()", title=None),
        )
        .properties(
            title=f"Average order value {locale.currency(df.Total.mean(), grouping=True)}"
        )
    )
    return fig


def plot_value_counts(series, title, scale="linear", bar_color="#5A5BC1"):
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

def plot_calendar_heatmap(df_calmap):
    df, year, last_sample_date = calendar_heatmap.wrangle_df(df_calmap, year=2023)
    return calendar_heatmap.plot_heatmap("Orders", df, legend=True, color_scheme="cividis")

def to_unordered_list(items):
    return f"""<ul style="margin:0;padding-left:20">{''.join([f'<li>{item}</li>' for item in items])}</ul>"""


def frequent_product_combinations(df_items_window):
    one_hot_encoded = (
        pd.get_dummies(df_items_window["Lineitem name"]).groupby("Name").sum()
    )

    # filter for only orders with 2 or more items
    one_hot_encoded_filtered = one_hot_encoded[one_hot_encoded.sum(axis=1) >= 2]

    frequent_itemsets = apriori(
        one_hot_encoded_filtered, min_support=0.025, use_colnames=True
    ).sort_values("support", ascending=False)
    assoc_rules = (
        association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
    )

    assoc_rules["Top Product Combinations"] = assoc_rules.apply(
        lambda row: to_unordered_list(
            sorted(list(row["antecedents"]) + list(row["consequents"]))
        ),
        axis=1,
    )

    frequent_combinations = (
        assoc_rules[["Top Product Combinations", "support"]]
        .drop_duplicates("Top Product Combinations", keep="first")
        .sort_values("support", ascending=False)
    )

    frequent_combinations["% of orders"] = round(
        frequent_combinations["support"] * 100, 2
    )

    frequent_combinations = frequent_combinations.drop(["support"], axis=1).reset_index(
        drop=True
    ).head(5)

    frequent_combinations.index = frequent_combinations.index + 1

    frequent_combinations = frequent_combinations.style.set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
            "vertical-align": "middle",
        }
    )

    frequent_combinations.format(na_rep="MISS", precision=2)

    cell_hover = { 
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
    }
    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #4340B1; color: white;'
    }
    frequent_combinations.set_table_styles([cell_hover, index_names, headers])

    return frequent_combinations
