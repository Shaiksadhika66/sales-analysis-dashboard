"""
Sales Analytics Dashboard
=========================
An interactive Streamlit + Plotly dashboard that runs instantly on a
synthetic, Superstore-style dataset generated on startup (no CSV needed).

Run with:
    pip install streamlit plotly pandas numpy
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_white"
CATEGORY_COLORS = {
    "Furniture": "#4C78A8",
    "Office Supplies": "#F58518",
    "Technology": "#54A24B",
}
SEGMENT_COLORS = {
    "Consumer": "#4C78A8",
    "Corporate": "#E45756",
    "Home Office": "#72B7B2",
}


def _stretch_kwargs() -> dict:
    """
    Streamlit >= 1.50 uses `width="stretch"`; older versions use
    `use_container_width=True`. Pick whichever the installed version supports
    so the script works across releases without deprecation errors.
    """
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 50):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()


# ------------------------------------------------------------------------------
# 1. DATA GENERATION
# ------------------------------------------------------------------------------
# Sub-category catalogue:
#   (category, sub-category, median unit price, price sigma, base margin, weight)
# Margins/prices loosely mimic the classic "Superstore" dataset: Technology is
# high-value and profitable, Furniture has thin (sometimes negative) margins,
# Office Supplies is high-volume and low-ticket.
SUBCATEGORIES = [
    ("Furniture", "Bookcases", 170, 0.45, 0.05, 0.05),
    ("Furniture", "Chairs", 180, 0.50, 0.13, 0.08),
    ("Furniture", "Furnishings", 38, 0.50, 0.20, 0.10),
    ("Furniture", "Tables", 210, 0.50, -0.04, 0.04),
    ("Office Supplies", "Appliances", 190, 0.70, 0.17, 0.05),
    ("Office Supplies", "Art", 18, 0.60, 0.22, 0.09),
    ("Office Supplies", "Binders", 28, 0.80, 0.34, 0.14),
    ("Office Supplies", "Envelopes", 20, 0.55, 0.32, 0.04),
    ("Office Supplies", "Fasteners", 9, 0.50, 0.15, 0.04),
    ("Office Supplies", "Labels", 12, 0.50, 0.42, 0.05),
    ("Office Supplies", "Paper", 22, 0.60, 0.40, 0.14),
    ("Office Supplies", "Storage", 75, 0.65, 0.12, 0.09),
    ("Office Supplies", "Supplies", 30, 0.80, -0.03, 0.04),
    ("Technology", "Accessories", 55, 0.70, 0.24, 0.07),
    ("Technology", "Copiers", 1100, 0.40, 0.30, 0.01),
    ("Technology", "Machines", 520, 0.50, 0.04, 0.02),
    ("Technology", "Phones", 210, 0.45, 0.14, 0.08),
]

# Region -> (sampling weight, profitability adjustment, [(state, city), ...])
REGIONS = {
    "West": (0.32, 0.02, [("California", "Los Angeles"), ("California", "San Francisco"),
                          ("California", "San Diego"), ("Washington", "Seattle"),
                          ("Oregon", "Portland")]),
    "East": (0.28, 0.01, [("New York", "New York City"), ("Pennsylvania", "Philadelphia"),
                          ("Massachusetts", "Boston"), ("Virginia", "Richmond")]),
    "Central": (0.23, -0.04, [("Texas", "Houston"), ("Texas", "Dallas"),
                              ("Illinois", "Chicago"), ("Michigan", "Detroit")]),
    "South": (0.17, 0.0, [("Florida", "Miami"), ("Georgia", "Atlanta"),
                          ("North Carolina", "Charlotte"), ("Tennessee", "Nashville")]),
}

FIRST_NAMES = ["Alex", "Maria", "James", "Priya", "Chen", "Olivia", "Liam", "Sofia", "Noah",
               "Emma", "Ravi", "Aisha", "Daniel", "Grace", "Lucas", "Hannah", "Omar", "Zoe",
               "Ethan", "Mei", "Carlos", "Nina", "Tom", "Fatima", "Jack", "Elena", "Sam", "Ivy"]
LAST_NAMES = ["Smith", "Johnson", "Patel", "Garcia", "Kim", "Brown", "Nguyen", "Davis", "Lopez",
              "Wilson", "Khan", "Martin", "Lee", "Clark", "Walker", "Young", "Allen", "King",
              "Scott", "Adams", "Baker", "Reed", "Cook", "Bell", "Murphy", "Rivera", "Cox"]
BRANDS = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Hooli", "Vandelay",
          "Soylent", "Wonka", "Cyberdyne", "Tyrell"]

SHIP_MODES = {  # mode -> (probability, min days, max days)
    "Standard Class": (0.60, 4, 7),
    "Second Class": (0.19, 2, 5),
    "First Class": (0.15, 1, 3),
    "Same Day": (0.06, 0, 0),
}


@st.cache_data(show_spinner="Generating sample data…")
def generate_data(
    n_orders: int = 5000,
    n_customers: int = 800,
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a realistic, reproducible, line-item level sales dataset."""
    rng = np.random.default_rng(seed)

    # --- Customers: segment, home location, and a skewed "activity" weight ----
    region_names = list(REGIONS)
    region_p = np.array([REGIONS[r][0] for r in region_names])
    cust_region = rng.choice(region_names, size=n_customers, p=region_p / region_p.sum())
    cust_loc = [
        REGIONS[r][2][rng.integers(len(REGIONS[r][2]))] for r in cust_region
    ]
    customers = pd.DataFrame({
        "Customer ID": [f"{rng.choice(list('ABCDEFGHJKLMNPRSTW'))}"
                        f"{rng.choice(list('ABCDEFGHJKLMNPRSTW'))}-{10000 + i}"
                        for i in range(n_customers)],
        "Customer Name": [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                          for _ in range(n_customers)],
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"],
                              size=n_customers, p=[0.52, 0.30, 0.18]),
        "Region": cust_region,
        "State": [s for s, _ in cust_loc],
        "City": [c for _, c in cust_loc],
        # Log-normal => a few loyal heavy buyers, many occasional ones
        "activity": rng.lognormal(mean=0.0, sigma=0.9, size=n_customers),
    })

    # --- Order dates: upward trend + Q4 holiday seasonality + weekday effect --
    days = pd.date_range(start, end, freq="D")
    t = np.linspace(0, 1, len(days))
    trend = 1.0 + 0.8 * t                                   # ~80% growth over the span
    month_season = np.array([0.80, 0.75, 1.00, 0.95, 1.00, 1.05,
                             0.95, 1.00, 1.30, 1.10, 1.45, 1.55])[days.month - 1]
    weekday = np.where(days.dayofweek >= 5, 0.65, 1.0)      # quieter weekends
    day_w = trend * month_season * weekday
    order_dates = rng.choice(days.to_numpy(), size=n_orders, p=day_w / day_w.sum())

    orders = pd.DataFrame({"Order Date": pd.to_datetime(order_dates)})
    orders = orders.sort_values("Order Date").reset_index(drop=True)
    orders["Order ID"] = [f"US-{d.year}-{i + 1:06d}" for i, d in enumerate(orders["Order Date"])]

    # Assign each order to a customer (weighted by activity)
    cust_p = customers["activity"] / customers["activity"].sum()
    orders["cust_idx"] = rng.choice(n_customers, size=n_orders, p=cust_p)

    # Shipping
    modes = list(SHIP_MODES)
    mode_p = np.array([SHIP_MODES[m][0] for m in modes])
    orders["Ship Mode"] = rng.choice(modes, size=n_orders, p=mode_p / mode_p.sum())
    lo = orders["Ship Mode"].map({m: v[1] for m, v in SHIP_MODES.items()}).to_numpy()
    hi = orders["Ship Mode"].map({m: v[2] for m, v in SHIP_MODES.items()}).to_numpy()
    ship_lag = rng.integers(lo, hi + 1)
    orders["Ship Date"] = orders["Order Date"] + pd.to_timedelta(ship_lag, unit="D")

    # --- Line items: 1–6 per order -------------------------------------------
    n_items = rng.choice([1, 2, 3, 4, 5, 6], size=n_orders, p=[0.35, 0.28, 0.18, 0.10, 0.06, 0.03])
    lines = orders.loc[orders.index.repeat(n_items)].reset_index(drop=True)
    n_lines = len(lines)

    # Product (sub-category) for each line
    sub = pd.DataFrame(SUBCATEGORIES, columns=["Category", "Sub-Category", "median", "sigma",
                                               "margin", "weight"])
    sub_idx = rng.choice(len(sub), size=n_lines, p=sub["weight"] / sub["weight"].sum())
    lines["Category"] = sub["Category"].to_numpy()[sub_idx]
    lines["Sub-Category"] = sub["Sub-Category"].to_numpy()[sub_idx]
    base_margin = sub["margin"].to_numpy()[sub_idx]

    # Product names (a small pool of "models" per sub-category)
    model_no = rng.integers(1, 13, size=n_lines)
    brand = np.array(BRANDS)[rng.integers(len(BRANDS), size=n_lines)]
    lines["Product Name"] = [f"{b} {s} Model {m}" for b, s, m in
                             zip(brand, lines["Sub-Category"], model_no)]
    lines["Product ID"] = [f"{c[:3].upper()}-{s[:2].upper()}-{10000 + m * 37 + i % 37}"
                           for i, (c, s, m) in enumerate(zip(lines["Category"],
                                                             lines["Sub-Category"], model_no))]

    # Attach customer attributes
    cust = customers.drop(columns="activity")
    lines = pd.concat([lines.reset_index(drop=True),
                       cust.iloc[lines["cust_idx"].to_numpy()].reset_index(drop=True)], axis=1)
    lines["Country"] = "United States"

    # --- Pricing, discounts, profit ------------------------------------------
    median_price = sub["median"].to_numpy()[sub_idx]
    unit_price = rng.lognormal(np.log(median_price), sub["sigma"].to_numpy()[sub_idx])
    unit_price = np.clip(unit_price, median_price * 0.2, median_price * 3.5)  # no absurd outliers
    quantity = rng.choice(np.arange(1, 11), size=n_lines,
                          p=[0.22, 0.24, 0.20, 0.14, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
    # Expensive items are rarely bought in bulk
    quantity = np.where(unit_price > 400, np.minimum(quantity, 3), quantity)
    # Central region discounts more often (which also hurts its margin)
    disc_levels = np.array([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    disc_default = np.array([0.55, 0.06, 0.05, 0.18, 0.07, 0.05, 0.04])
    disc_central = np.array([0.30, 0.05, 0.05, 0.30, 0.12, 0.10, 0.08])
    u = rng.random(n_lines)
    cdf_default, cdf_central = np.cumsum(disc_default), np.cumsum(disc_central)
    is_central = (lines["Region"] == "Central").to_numpy()
    idx_default = np.searchsorted(cdf_default, u).clip(max=len(disc_levels) - 1)
    idx_central = np.searchsorted(cdf_central, u).clip(max=len(disc_levels) - 1)
    discount = disc_levels[np.where(is_central, idx_central, idx_default)]

    region_adj = lines["Region"].map({r: v[1] for r, v in REGIONS.items()}).to_numpy()
    margin = base_margin + region_adj - 0.45 * discount + rng.normal(0, 0.04, n_lines)

    sales = unit_price * quantity * (1 - discount)
    lines["Quantity"] = quantity
    lines["Discount"] = discount
    lines["Sales"] = sales.round(2)
    lines["Profit"] = (sales * margin).round(2)

    columns = [
        "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name",
        "Segment", "Country", "City", "State", "Region", "Product ID", "Category",
        "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit",
    ]
    return lines[columns].sort_values(["Order Date", "Order ID"]).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 2. ANALYTICS HELPERS (pure functions — easy to unit-test)
# ------------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, regions, categories, start, end) -> pd.DataFrame:
    """Filter by region, category and an inclusive [start, end] date range."""
    mask = (
        df["Region"].isin(regions)
        & df["Category"].isin(categories)
        & (df["Order Date"] >= pd.Timestamp(start))
        & (df["Order Date"] <= pd.Timestamp(end))
    )
    return df.loc[mask]


def compute_kpis(df: pd.DataFrame) -> dict:
    """Headline KPIs for a slice of the data."""
    orders = df["Order ID"].nunique()
    sales = float(df["Sales"].sum())
    return {
        "sales": sales,
        "profit": float(df["Profit"].sum()),
        "aov": sales / orders if orders else 0.0,   # average order value
        "customers": int(df["Customer ID"].nunique()),
    }


def pct_delta(current: float, previous: float) -> str | None:
    """Format a % change vs the previous period (None when it can't be computed)."""
    if previous is None or previous == 0:
        return None
    return f"{(current - previous) / abs(previous) * 100:+.1f}%"


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# ------------------------------------------------------------------------------
# 3. LOAD DATA + SIDEBAR FILTERS
# ------------------------------------------------------------------------------
df_all = generate_data()

st.sidebar.title("🔎 Filters")

all_regions = sorted(df_all["Region"].unique())
all_categories = sorted(df_all["Category"].unique())

sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)

min_date = df_all["Order Date"].min().date()
max_date = df_all["Order Date"].max().date()
start_date, end_date = st.sidebar.slider(
    "Order date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

st.sidebar.caption(
    "KPI deltas compare the selected range with the equally long period "
    "immediately before it."
)

# ------------------------------------------------------------------------------
# 4. HEADER + GUARD AGAINST EMPTY SELECTIONS
# ------------------------------------------------------------------------------
st.title("📊 Sales Analytics Dashboard")
st.caption(f"Showing orders from **{start_date:%b %d, %Y}** to **{end_date:%b %d, %Y}**")

if not sel_regions or not sel_categories:
    st.warning("Select at least one Region and one Category in the sidebar.")
    st.stop()

df = apply_filters(df_all, sel_regions, sel_categories, start_date, end_date)
if df.empty:
    st.info("No orders match the current filters. Try widening the date range.")
    st.stop()

# ------------------------------------------------------------------------------
# 5. KEY METRICS (with delta vs previous period)
# ------------------------------------------------------------------------------
cur = compute_kpis(df)

# Previous period = same-length window directly before the selected range,
# with the same Region/Category filters.
span = pd.Timestamp(end_date) - pd.Timestamp(start_date) + pd.Timedelta(days=1)
prev_start = pd.Timestamp(start_date) - span
prev_end = pd.Timestamp(start_date) - pd.Timedelta(days=1)
prev = None
if prev_start >= df_all["Order Date"].min():   # only if the prior window has full data
    prev_df = apply_filters(df_all, sel_regions, sel_categories, prev_start, prev_end)
    if not prev_df.empty:
        prev = compute_kpis(prev_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sales", fmt_money(cur["sales"]),
          pct_delta(cur["sales"], prev["sales"]) if prev else None)
m2.metric("Total Profit", fmt_money(cur["profit"]),
          pct_delta(cur["profit"], prev["profit"]) if prev else None,
          help=f"Profit margin: {cur['profit'] / cur['sales']:.1%}")
m3.metric("Avg. Order Value", f"${cur['aov']:,.2f}",
          pct_delta(cur["aov"], prev["aov"]) if prev else None)
m4.metric("Customers", f"{cur['customers']:,}",
          pct_delta(cur["customers"], prev["customers"]) if prev else None)

if prev is None:
    st.caption("ℹ️ Deltas are hidden because there isn't a full prior period in the data.")

st.divider()

# ------------------------------------------------------------------------------
# 6. VISUALIZATIONS
# ------------------------------------------------------------------------------
# 6a. Time-series trend ---------------------------------------------------------
st.subheader("📈 Trend over time")
c1, c2, _ = st.columns([1, 1, 2])
metric_choice = c1.selectbox("Metric", ["Sales", "Profit"])
grain = c2.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[grain]
ma_window = {"Daily": 14, "Weekly": 4, "Monthly": 3}[grain]

ts = (
    df.set_index("Order Date")[metric_choice]
    .resample(freq).sum()
    .rename(metric_choice).reset_index()
)
ts[f"{ma_window}-period moving avg"] = ts[metric_choice].rolling(ma_window, min_periods=1).mean()

fig_ts = px.line(
    ts, x="Order Date", y=[metric_choice, f"{ma_window}-period moving avg"],
    template=PLOTLY_TEMPLATE, labels={"value": metric_choice, "variable": ""},
    color_discrete_sequence=["#4C78A8", "#E45756"],
)
fig_ts.update_traces(selector=dict(name=f"{ma_window}-period moving avg"), line=dict(dash="dash"))
fig_ts.update_yaxes(tickprefix="$", separatethousands=True)
fig_ts.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0),
                     margin=dict(l=10, r=10, t=30, b=10), height=380)
st.plotly_chart(fig_ts, **STRETCH)

# 6b + 6c: category bar and segment breakdown side by side -------------------
left, right = st.columns(2)

with left:
    st.subheader("🏷️ Category performance")
    level = st.radio("Group by", ["Category", "Sub-Category"], horizontal=True)
    perf = (
        df.groupby(level, as_index=False)[["Sales", "Profit"]].sum()
        .sort_values("Sales", ascending=False)
    )
    perf_long = perf.melt(id_vars=level, value_vars=["Sales", "Profit"],
                          var_name="Measure", value_name="Amount")
    fig_bar = px.bar(
        perf_long, x=level, y="Amount", color="Measure", barmode="group",
        template=PLOTLY_TEMPLATE,
        color_discrete_map={"Sales": "#4C78A8", "Profit": "#54A24B"},
    )
    fig_bar.update_yaxes(tickprefix="$", separatethousands=True)
    fig_bar.update_layout(legend=dict(orientation="h", y=1.1, x=0), xaxis_title=None,
                          margin=dict(l=10, r=10, t=30, b=10), height=400)
    st.plotly_chart(fig_bar, **STRETCH)

with right:
    st.subheader("👥 Customer segments")
    view = st.radio("View", ["Sales share (donut)", "Customers (scatter)"], horizontal=True)
    if view.startswith("Sales"):
        seg = df.groupby("Segment", as_index=False)["Sales"].sum()
        fig_seg = px.pie(
            seg, names="Segment", values="Sales", hole=0.45,
            template=PLOTLY_TEMPLATE, color="Segment", color_discrete_map=SEGMENT_COLORS,
        )
        fig_seg.update_traces(textinfo="percent+label", textposition="inside")
        fig_seg.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), height=400)
    else:
        cust = (
            df.groupby(["Customer ID", "Customer Name", "Segment"], as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                 Orders=("Order ID", "nunique"))
        )
        fig_seg = px.scatter(
            cust, x="Sales", y="Profit", size="Orders", color="Segment",
            hover_name="Customer Name", hover_data={"Customer ID": True, "Orders": True},
            template=PLOTLY_TEMPLATE, color_discrete_map=SEGMENT_COLORS,
            size_max=22, opacity=0.7,
        )
        fig_seg.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_seg.update_xaxes(tickprefix="$", separatethousands=True)
        fig_seg.update_yaxes(tickprefix="$", separatethousands=True)
        fig_seg.update_layout(legend=dict(orientation="h", y=1.1, x=0),
                              margin=dict(l=10, r=10, t=30, b=10), height=400)
    st.plotly_chart(fig_seg, **STRETCH)

st.divider()

# ------------------------------------------------------------------------------
# 7. DATA VIEW & DOWNLOAD
# ------------------------------------------------------------------------------
st.subheader("🗂️ Filtered data")

top_l, top_r = st.columns([3, 1])
top_l.caption(f"{len(df):,} line items across {df['Order ID'].nunique():,} orders "
              "— click a column header to sort.")

csv_bytes = df.to_csv(index=False).encode("utf-8")
top_r.download_button(
    label="⬇️ Download CSV",
    data=csv_bytes,
    file_name=f"sales_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
    mime="text/csv",
)

st.dataframe(
    df,
    hide_index=True,
    height=420,
    column_config={
        "Order Date": st.column_config.DateColumn("Order Date", format="YYYY-MM-DD"),
        "Ship Date": st.column_config.DateColumn("Ship Date", format="YYYY-MM-DD"),
        "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
        "Profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
        "Discount": st.column_config.NumberColumn("Discount", format="%.2f"),
    },
    **STRETCH,
)
