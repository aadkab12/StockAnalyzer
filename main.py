import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import warnings
from datetime import datetime

# Suppress warnings for a clean terminal output
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("📈 Pro Quant Terminal & Screener")


# --- 2. ROBUST NEWS SCRAPER (GOOGLE NEWS RSS) ---
def _normalize_yf_news(yf_items, limit):
    headlines = []
    for item in yf_items:
        title = ""
        url = ""
        publisher = "Yahoo Finance"
        date_str = "Unknown date"

        # Legacy yfinance schema.
        if isinstance(item, dict):
            title = item.get("title", "") or ""
            url = item.get("link", "") or ""
            publisher = item.get("publisher", publisher) or publisher
            provider_time = item.get("providerPublishTime")
            if provider_time:
                date_str = datetime.fromtimestamp(provider_time).strftime("%Y-%m-%d %H:%M")

            # Newer yfinance schema with nested "content".
            content = item.get("content", {}) if isinstance(item.get("content", {}), dict) else {}
            if content:
                title = content.get("title", title) or title
                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    url = canonical.get("url", url) or url
                pub_date = content.get("pubDate")
                if isinstance(pub_date, str) and pub_date.strip():
                    date_str = pub_date
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", publisher) or publisher

        # Prevent empty/placeholder rows in UI.
        if not title.strip() or not url.strip():
            continue

        headlines.append(
            {
                "title": title.strip(),
                "link": url.strip(),
                "date": date_str,
                "source": publisher,
            }
        )

        if len(headlines) >= limit:
            break
    return headlines


def _fetch_google_rss(ticker, topic, limit):
    query = f"{ticker} {topic}".strip()
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    headlines = []

    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        if not title or not link:
            continue
        headlines.append(
            {
                "title": title,
                "link": link,
                "date": item.findtext("pubDate", default=""),
                "source": item.findtext("source", default="Google News"),
            }
        )
    return headlines


def _fetch_yahoo_rss(ticker, limit):
    encoded_ticker = urllib.parse.quote_plus(ticker)
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={encoded_ticker}&region=US&lang=en-US"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    headlines = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        if not title or not link:
            continue
        headlines.append(
            {
                "title": title,
                "link": link,
                "date": item.findtext("pubDate", default=""),
                "source": "Yahoo Finance RSS",
            }
        )
    return headlines


def fetch_latest_news(ticker, topic="stock", limit=8):
    """Returns ticker-specific headlines using a fallback chain."""
    errors = []

    try:
        yf_news = yf.Ticker(ticker).news
        if yf_news:
            return _normalize_yf_news(yf_news, limit), "yfinance", None
    except Exception as e:
        errors.append(f"yfinance failed: {e}")

    try:
        google_news = _fetch_google_rss(ticker, topic, limit)
        if google_news:
            return google_news, "google_rss", None
    except Exception as e:
        errors.append(f"Google RSS failed: {e}")

    try:
        yahoo_news = _fetch_yahoo_rss(ticker, limit)
        if yahoo_news:
            return yahoo_news, "yahoo_rss", None
    except Exception as e:
        errors.append(f"Yahoo RSS failed: {e}")

    return [], "none", " | ".join(errors) if errors else "No providers returned headlines."


# --- 3. DATA & QUANT ENGINE (NATIVE PANDAS) ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_analyze_data(ticker, period="1y"):
    try:
        # Fetch Stock and Benchmark (SPY) Data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        spy = yf.download('SPY', period=period, progress=False)['Close']

        if df.empty or len(df) < 60:
            return None, None, None

        # Clean yfinance multi-index if necessary
        if isinstance(spy, pd.DataFrame):
            spy = spy.squeeze()

        # 1. Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # 2. RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # 3. ATR (14-day)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

        # 4. Average Daily Volume (20-day)
        df['ADV_20'] = df['Volume'].rolling(window=20).mean()

        # Calculate Beta
        stock_returns = df['Close'].pct_change().dropna()
        spy_returns = spy.pct_change().dropna()
        aligned_returns = pd.concat([stock_returns, spy_returns], axis=1, join='inner').dropna()
        aligned_returns.columns = ['Stock', 'SPY']

        covariance = np.cov(aligned_returns['Stock'], aligned_returns['SPY'])[0][1]
        variance = np.var(aligned_returns['SPY'])
        beta = covariance / variance if variance != 0 else 0

        info = stock.info
        return df, info, beta
    except Exception:
        return None, None, None


# --- 4. UI AND LAYOUT ---
st.sidebar.header("Terminal Controls")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", "XOM").upper()
timeframe = st.sidebar.selectbox("Chart Timeframe", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Live News Settings")
news_topic = st.sidebar.selectbox(
    "News focus",
    ["stock", "earnings", "guidance", "analyst", "merger", "lawsuit"],
    index=0,
)
news_limit = st.sidebar.slider("Max headlines", min_value=5, max_value=20, value=10, step=1)
auto_refresh = st.sidebar.toggle("Auto-refresh news", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 15, 300, 60, 15)
refresh_clicked = st.sidebar.button("Refresh Now")

if refresh_clicked:
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    # Lightweight browser-side timer to keep data and news current without extra dependencies.
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_seconds * 1000});
        </script>
        """,
        height=0,
        width=0,
    )
    st.sidebar.caption(f"Auto-refresh active: every {refresh_seconds} seconds")

if ticker_input:
    with st.spinner(f"Pulling live data and calculating metrics for {ticker_input}..."):
        df, info, beta = load_and_analyze_data(ticker_input, timeframe)
        news_data, news_provider, news_error = fetch_latest_news(ticker_input, topic=news_topic, limit=news_limit)

    if df is None:
        st.error(f"Error fetching data for {ticker_input}. Ensure the ticker is valid.")
    else:
        current = df.iloc[-1]

        # --- Top Row: Fundamentals & Strategy Checklist ---
        st.subheader(f"{info.get('shortName', ticker_input)} ({ticker_input})")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            "Current Price",
            f"${current['Close']:.2f}",
            f"{(current['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100:.2f}%",
        )
        col2.metric("RSI (14)", f"{current['RSI_14']:.2f}")
        col3.metric("Beta", f"{beta:.2f}")
        col4.metric("ATR (14)", f"${current['ATR_14']:.2f}")
        col5.metric("Avg Daily Vol (20)", f"{current['ADV_20'] / 1e6:.2f}M")

        st.markdown("---")

        # --- Middle Row: Chart & News ---
        chart_col, news_col = st.columns([2.5, 1])

        with chart_col:
            st.markdown("**Technical Chart (Price + 20 EMA & 50 SMA)**")
            fig = go.Figure()

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price",
                )
            )
            # 50 SMA
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1.5), name="50 SMA")
            )
            # 20 EMA
            fig.add_trace(
                go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1.5), name="20 EMA")
            )

            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                height=550,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Automated Strategy Check
            st.markdown("### 🚦 Strategy Checklist Status")
            checks = {
                "Liquidity (ADV > 500k)": current['ADV_20'] > 500000,
                "Volatility (Beta > 1)": beta > 1,
                "Trend (Price > 50 SMA)": current['Close'] > current['SMA_50'],
                "Momentum (RSI > 50)": current['RSI_14'] > 50,
                "Bullish Crossover (20 EMA > 50 SMA)": current['EMA_20'] > current['SMA_50'],
            }

            for check, passed in checks.items():
                icon = "✅ Pass" if passed else "❌ Fail"
                st.write(f"**{icon}** | {check}")

        with news_col:
            st.markdown("**Live Catalyst Feed (RSS)**")
            st.caption(f"Query: {ticker_input} + {news_topic}")
            st.caption(f"Provider: {news_provider}")
            st.caption(f"Headlines loaded: {len(news_data)}")
            if news_data:
                for article in news_data:
                    st.markdown(f"[{article['title']}]({article['link']})")
                    st.caption(f"{article['source']} | {article['date']}")
                    st.markdown("---")
            else:
                st.write("No recent catalyst news available.")
                if news_error:
                    st.warning(news_error)
