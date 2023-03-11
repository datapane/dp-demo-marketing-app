## Preparing the data

If you'd like to reproduce the data in the parent directory, i.e. `cust.csv`, `items.csv`, and `order.csv`, you can with the following:

1. Retrieve `cust_pub4.csv` from the data source below and place it in this directory.
2. Run `pydata_adjustments.ipynb` which outputs `pydata_adjustments.csv` in this directory.
3. Run `Cap2_cust_datawrangling2_v2.ipynb` which outputs the three CSV files in the parent directory.

## Adjustments to the data

We made the following adjustments to the data:

1. Added three years to the date columns to make the data appear to be more recent.
2. Dropped any rows that were beyond the PyData presentation slot, i.e. 2023-04-27 10:15.

## Data source

Data was obtained from https://github.com/sjrekuc/Shopify-CLV, the original README is below.

> # Capstone2_Purchase
> ========================= <br>
> Revenue growth is great in a young business, but to sustain that growth, the business must retain customers as repeat buyers. Acquiring new customers requires significantly more money in marketing than to just retain the existing customers. A valuable company will be able to retain customers that try their products and therefore sustain their revenue growth. We model a small online business that sells through Shopify. We focus on characteristics from a customer’s first purchase that could indicate their Customer Lifetime Value (CLV) and we model that CLV from the features obtained in the first purchase.
>
> ## Project Organization
>
> -------------- <br>
> - LICENSE <br>
> - README.md     <- Top-Level Readme for anyone > interested in this project <br>
> - DATA <br>
>  - Public Raw < - Original CSV anonymized customer data from Shopify <br>
>    - cust_pub4.pkl < - pickled python DataFrame after removing all identifying information about customers
>    - cust_pub4.csv < - CSV file of customer DataFrame
