import locale
import typing as t

import math

import altair as alt
import numpy as np
import pandas as pd
from dominate.tags import *
import datapane as dp

import analytics as a


def gen_cohort_analysis(
    df_orders_window: pd.DataFrame,
) -> dp.Group:
    df_orders_cohort = df_orders_window

    df_orders_cohort["order_month"] = df_orders_cohort["Created at"].apply(a.get_month)

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
                alt.Tooltip("cohort_index:O", title="Cohort Index"),
                alt.Tooltip("cohort_month:O", title="Cohort Month"),
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

    return dp.Group(
        dp.Plot(retention_chart),
        cohort_data,
        dp.Plot(avg_order_chart),
        average_standard_cost,
        columns=2,
        label="Retention analysis",
    )
