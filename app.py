
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="StockPilot AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 StockPilot AI")
st.caption(
    "Malaysian stock analysis and AI-powered annual report research"
)

# --------------------------------------------------
# Stock CSV configuration
# Adjust filenames if yours are different.
# --------------------------------------------------

STOCK_FILES = {
    "Maybank": "Maybank.csv",
    "MR D.I.Y.": "MRDIY.csv",
    "Tenaga Nasional": "Tenaga_Nasional.csv",
}

TICKERS = {
    "Maybank": "1155.KL",
    "MR D.I.Y.": "5296.KL",
    "Tenaga Nasional": "5347.KL",
}


# --------------------------------------------------
# Load and normalize historical stock CSVs
# Supports ordinary CSVs and common yfinance
# multi-header CSV exports.
# --------------------------------------------------

def find_csv(filename):
    candidates = [
        Path("assets") / filename,
        Path(filename),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


@st.cache_data(show_spinner=False)
def load_stock_csv(path_string, modified_time):
    path = Path(path_string)
    raw = pd.read_csv(path)

    # Detect common yfinance multi-level CSV exports.
    first_value = ""
    if not raw.empty:
        first_value = str(raw.iloc[0, 0]).strip().lower()

    known_price_columns = {
        str(column).strip().lower()
        for column in raw.columns
    }

    is_multi_header = (
        first_value in {"ticker", "date", "price"}
        and bool(
            known_price_columns
            & {"close", "high", "low", "open", "adj close", "volume"}
        )
    )

    if is_multi_header:
        df = pd.read_csv(path, header=[0, 1])

        # Flatten the two-level headings.
        flattened = []
        for column in df.columns:
            first, second = str(column[0]), str(column[1])

            if (
                first.lower() in {"price", "ticker"}
                or first.lower().startswith("unnamed")
            ):
                flattened.append(second)
            else:
                flattened.append(first)

        df.columns = flattened

        # Remove the extra "Date" row present in some yfinance exports.
        first_column = df.columns[0]
        values = df[first_column].astype(str).str.strip().str.lower()
        df = df.loc[~values.isin({"date", "ticker", "price"})].copy()

        if df.columns[0].strip().lower() in {
            "ticker", "price", "unnamed: 0"
        }:
            df = df.rename(columns={df.columns[0]: "Date"})

    else:
        df = raw.copy()

        # Handle CSVs where the date index was saved as an unnamed column.
        if "Date" not in df.columns:
            possible_date = [
                column for column in df.columns
                if str(column).strip().lower()
                in {"datetime", "timestamp", "unnamed: 0"}
            ]
            if possible_date:
                df = df.rename(columns={possible_date[0]: "Date"})

    df.columns = [str(column).strip() for column in df.columns]

    # Find the date column.
    date_column = next(
        (
            column for column in df.columns
            if column.strip().lower() in {"date", "datetime", "timestamp"}
        ),
        None,
    )

    if date_column is None and len(df.columns):
        date_column = df.columns[0]
        df = df.rename(columns={date_column: "Date"})
        date_column = "Date"

    # Prefer the ordinary closing price over adjusted close.
    close_column = next(
        (
            column for column in df.columns
            if column.strip().lower() == "close"
        ),
        None,
    )

    if close_column is None:
        close_column = next(
            (
                column for column in df.columns
                if column.strip().lower() == "adj close"
            ),
            None,
        )

    if close_column is None:
        raise ValueError(
            f"No Close or Adj Close column found in {path.name}. "
            f"Columns found: {df.columns.tolist()}"
        )

    df["Date"] = pd.to_datetime(df[date_column], errors="coerce")
    if date_column != "Date":
        df = df.drop(columns=[date_column])

    # Normalize price and volume columns.
    rename_map = {}
    for column in df.columns:
        normalized = column.strip().lower()
        if normalized == "close":
            rename_map[column] = "Close"
        elif normalized == "adj close":
            rename_map[column] = "Adj Close"
        elif normalized == "open":
            rename_map[column] = "Open"
        elif normalized == "high":
            rename_map[column] = "High"
        elif normalized == "low":
            rename_map[column] = "Low"
        elif normalized == "volume":
            rename_map[column] = "Volume"

    df = df.rename(columns=rename_map)

    if "Close" not in df.columns:
        df = df.rename(columns={close_column: "Close"})

    for column in [
        "Close", "Adj Close", "Open", "High", "Low", "Volume"
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return df


# --------------------------------------------------
# Load available datasets
# --------------------------------------------------

stock_data = {}
missing_files = []

for company, filename in STOCK_FILES.items():
    path = find_csv(filename)

    if path is None:
        missing_files.append(
            f"{company}: expected {filename} in the project "
            "root or assets/ folder."
        )
        continue

    try:
        stock_data[company] = load_stock_csv(
            str(path.resolve()),
            path.stat().st_mtime,
        )
    except Exception as exc:
        st.error(f"Couldn't load {company}'s CSV: {exc}")

if missing_files:
    with st.expander("Check your stock CSV files"):
        for message in missing_files:
            st.warning(message)

if not stock_data:
    st.error(
        "No stock data could be loaded. Check your CSV filenames "
        "and confirm that they contain Date and Close columns."
    )
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Dashboard settings")

available_companies = list(stock_data.keys())

selected_companies = st.sidebar.multiselect(
    "Companies to compare",
    options=available_companies,
    default=available_companies,
)

if not selected_companies:
    st.info("Select at least one company in the sidebar.")
    st.stop()


# --------------------------------------------------
# Dashboard tab
# --------------------------------------------------

dashboard_tab, ai_tab = st.tabs([
    "📊 Stock Dashboard",
    "💬 Annual Report AI",
])

with dashboard_tab:
    st.subheader("Market overview")

    metric_columns = st.columns(len(selected_companies))

    for column, company in zip(metric_columns, selected_companies):
        df = stock_data[company]
        latest_price = float(df["Close"].iloc[-1])
        latest_date = df["Date"].iloc[-1]

        # Approximate one-year return using 252 trading observations.
        comparison_index = max(0, len(df) - 252)
        reference_price = float(df["Close"].iloc[comparison_index])

        if reference_price:
            yearly_return = (
                latest_price / reference_price - 1
            ) * 100
        else:
            yearly_return = 0.0

        daily_return = None
        if len(df) >= 2:
            previous_price = float(df["Close"].iloc[-2])
            if previous_price:
                daily_return = (
                    latest_price / previous_price - 1
                ) * 100

        with column:
            st.metric(
                label=company,
                value=f"RM {latest_price:,.3f}",
                delta=(
                    f"{daily_return:+.2f}% daily"
                    if daily_return is not None
                    else None
                ),
            )
            st.caption(
                f"{TICKERS.get(company, '')} · "
                f"{latest_date:%d %b %Y}"
            )
            st.caption(
                f"Approx. 1-year return: {yearly_return:+.2f}%"
            )

    st.divider()

    # Individual price chart.
    chart_company = st.selectbox(
        "View historical price",
        options=selected_companies,
    )

    chart_df = stock_data[chart_company].copy()

    st.subheader(f"{chart_company} — closing price")

    price_fig = px.line(
        chart_df,
        x="Date",
        y="Close",
        labels={
            "Date": "Date",
            "Close": "Closing price (RM)",
        },
    )
    price_fig.update_layout(
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # Calculate moving averages when sufficient observations exist.
    chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
    chart_df["MA50"] = chart_df["Close"].rolling(50).mean()

    st.subheader("Moving averages")

    ma_fig = px.line(
        chart_df,
        x="Date",
        y=["Close", "MA20", "MA50"],
        labels={
            "Date": "Date",
            "value": "Price (RM)",
            "variable": "Series",
        },
    )
    ma_fig.update_layout(
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(ma_fig, use_container_width=True)

    # Normalize each selected stock to 100 at its first observation.
    st.subheader("Performance comparison")

    comparison_frames = []

    for company in selected_companies:
        df = stock_data[company][["Date", "Close"]].copy()

        if df.empty or df["Close"].iloc[0] == 0:
            continue

        df["Indexed performance"] = (
            df["Close"] / df["Close"].iloc[0]
        ) * 100
        df["Company"] = company

        comparison_frames.append(
            df[["Date", "Indexed performance", "Company"]]
        )

    if comparison_frames:
        comparison_df = pd.concat(
            comparison_frames,
            ignore_index=True,
        )

        comparison_fig = px.line(
            comparison_df,
            x="Date",
            y="Indexed performance",
            color="Company",
            labels={
                "Date": "Date",
                "Indexed performance": "Performance index (start = 100)",
            },
        )
        comparison_fig.update_layout(
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
        )

        st.caption(
            "Each stock is normalized to 100 on its own first date. "
            "The comparison periods may therefore differ."
        )

    # Historical data table.
    with st.expander(f"View {chart_company} historical data"):
        st.dataframe(
            chart_df.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# Annual report RAG chatbot
# --------------------------------------------------

with ai_tab:
    st.subheader("Ask questions about the 2025 annual reports")

    st.write(
        "Ask about Maybank, MR D.I.Y., or Tenaga Nasional. "
        "Answers are generated from passages retrieved from your "
        "Azure AI Search index."
    )

    try:
        from stock_agent import answer_question
        ai_ready = True
    except Exception as exc:
        ai_ready = False
        st.error(
            "The AI assistant couldn't be loaded. "
            "Check stock_agent.py, your .env configuration, "
            "and installed dependencies."
        )
        st.exception(exc)

    company_scope = st.selectbox(
        "Report scope",
        [
            "All companies",
            "Maybank",
            "MR D.I.Y.",
            "Tenaga Nasional",
        ],
    )

    with st.form("annual_report_question"):
        question = st.text_area(
            "Your question",
            placeholder=(
                "For example: What were Maybank's principal risks "
                "in 2025?"
            ),
            height=100,
        )

        submitted = st.form_submit_button(
            "Ask StockPilot AI",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        elif ai_ready:
            scoped_question = question.strip()

            if company_scope != "All companies":
                scoped_question = (
                    f"Focus on {company_scope}'s 2025 annual report. "
                    f"Question: {scoped_question}"
                )

            try:
                with st.spinner(
                    "Searching the annual reports and generating an answer..."
                ):
                    answer, sources = answer_question(scoped_question)

                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown("### Retrieved sources")

                if sources:
                    for source in sources:
                        reference = source.get("reference", "?")
                        company = source.get("company", "Unknown company")
                        filename = source.get("source", "Unknown file")
                        page = source.get("page", "?")

                        with st.expander(
                            f"[{reference}] {company} — PDF page {page}"
                        ):
                            st.write(f"**File:** {filename}")
                            st.write(f"**Page:** {page}")

                            score = source.get("score")
                            if score is not None:
                                st.write(
                                    f"**Search score:** {score:.4f}"
                                )
                else:
                    st.info("No source passages were returned.")

                st.caption(
                    "PDF page numbers may differ from the printed "
                    "page numbers in the reports."
                )

            except Exception as exc:
                st.error(f"Couldn't answer the question: {exc}")
                st.info(
                    "Check that your chat deployment, embedding "
                    "deployment, Search endpoint, key, and index "
                    "name are correct in .env."
                )

    st.divider()
    st.caption(
        "StockPilot AI is a research tool, not personalized "
        "investment advice. Historical performance does not "
        "guarantee future returns."
    )