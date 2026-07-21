import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from forecast import forecast_prices
# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Corn Futures AI Trader",
    page_icon="🌽",
    layout="wide"
)
# ==========================
# PAPER TRADING SESSION
# ==========================

if "cash" not in st.session_state:
    st.session_state.cash = 100000.0

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

if "transactions" not in st.session_state:
    st.session_state.transactions = []
# ── STYLES ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .signal-box {
        padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px;
        font-weight: bold; margin: 10px 0;
    }
    .buy  { background-color: #1a4731; color: #00ff88; border: 2px solid #00ff88; }
    .sell { background-color: #4a1a1a; color: #ff4444; border: 2px solid #ff4444; }
    .hold { background-color: #2a2a1a; color: #ffcc00; border: 2px solid #ffcc00; }
</style>
""", unsafe_allow_html=True)
# ==========================
# PAPER TRADING SIDEBAR
# ==========================

st.sidebar.header("💼 Paper Trading")

st.sidebar.metric(
    "Available Cash",
    f"${st.session_state.cash:,.2f}"
)
# ── HEADER ────────────────────────────────────────────────────
st.title("🌽 Corn Futures AI Trading Agent")
st.write(f"💰 Cash: ${st.session_state.cash:,.2f}")
st.markdown("*Multiscale candlestick analysis powered by machine learning*")
st.divider()

# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("ZC=F", period="2y", interval="1d", progress=False)
    df.columns = [col[0] for col in df.columns]
    df_h = yf.download("ZC=F", period="730d", interval="1h", progress=False)
    df_h.columns = [col[0] for col in df_h.columns]
    return df, df_h

@st.cache_resource
def train_model(df, df_h):
    # Daily features
    d = df.copy()
    d['body']          = d['Close'] - d['Open']
    d['body_pct']      = (d['body'] / d['Open']) * 100
    d['upper_wick']    = d['High'] - d[['Open','Close']].max(axis=1)
    d['lower_wick']    = d[['Open','Close']].min(axis=1) - d['Low']
    d['direction']     = (d['Close'] > d['Open']).astype(int)
    d['ma_10']         = d['Close'].rolling(10).mean()
    d['ma_20']         = d['Close'].rolling(20).mean()
    d['ma_50']         = d['Close'].rolling(50).mean()
    d['above_ma10']    = (d['Close'] > d['ma_10']).astype(int)
    d['above_ma50']    = (d['Close'] > d['ma_50']).astype(int)
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    d['rsi']           = 100 - (100 / (1 + gain / loss))
    d['rsi_change']    = d['rsi'].diff()
    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['macd']          = ema12 - ema26
    d['macd_signal']   = d['macd'].ewm(span=9).mean()
    d['macd_hist']     = d['macd'] - d['macd_signal']
    hl = d['High'] - d['Low']
    hc = (d['High'] - d['Close'].shift()).abs()
    lc = (d['Low']  - d['Close'].shift()).abs()
    d['atr']           = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    d['volume_spike']  = (d['Volume'] > d['Volume'].rolling(20).mean() * 1.5).astype(int)
    d['return_1d']     = d['Close'].pct_change(1)  * 100
    d['return_3d']     = d['Close'].pct_change(3)  * 100
    d['return_5d']     = d['Close'].pct_change(5)  * 100
    d['return_10d']    = d['Close'].pct_change(10) * 100
    d['high_14']       = d['High'].rolling(14).max()
    d['low_14']        = d['Low'].rolling(14).min()
    d['price_position']= (d['Close'] - d['low_14']) / (d['high_14'] - d['low_14'])
    d['is_doji']       = (d['body_pct'].abs() < 0.1).astype(int)
    d['streak']        = d['direction'].groupby(
        (d['direction'] != d['direction'].shift()).cumsum()
    ).cumcount() + 1
    d.loc[d['direction'] == 0, 'streak'] *= -1

    # Hourly features
    df_h['h_body_pct']   = (df_h['Close'] - df_h['Open']) / df_h['Open'] * 100
    dh = df_h['Close'].diff()
    gh = dh.clip(lower=0).rolling(14).mean()
    lh = -dh.clip(upper=0).rolling(14).mean()
    df_h['h_rsi']        = 100 - (100 / (1 + gh / lh))
    df_h['h_ma_10']      = df_h['Close'].rolling(10).mean()
    df_h['h_above_ma10'] = (df_h['Close'] > df_h['h_ma_10']).astype(int)
    df_h['h_volatility'] = df_h['Close'].rolling(10).std()
    dh_daily = df_h[['h_body_pct','h_rsi','h_above_ma10','h_volatility']].resample('D').last()
    dh_daily.index = dh_daily.index.tz_localize(None)

    # Target
    fr = (d['Close'].shift(-5) - d['Close']) / d['Close'] * 100
    d['target'] = 1
    d.loc[fr >  1.0, 'target'] = 2
    d.loc[fr < -1.0, 'target'] = 0
    d = d.iloc[:-5]

    d.index = d.index.tz_localize(None)
    dm = d.merge(dh_daily, left_index=True, right_index=True, how='inner').dropna()

    features = [
        'body','body_pct','upper_wick','lower_wick','direction',
        'ma_10','ma_20','ma_50','above_ma10','above_ma50',
        'rsi','rsi_change','macd','macd_signal','macd_hist',
        'atr','volume_spike','return_1d','return_3d','return_5d',
        'return_10d','price_position','is_doji','streak',
        'h_body_pct','h_rsi','h_above_ma10','h_volatility'
    ]

    X = dm[features].values
    y = dm['target'].values
    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_all   = scaler.transform(X)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y[:split])
    return model, scaler, features, dm

# ── LOAD ──────────────────────────────────────────────────────
with st.spinner("Loading corn futures data..."):
    df, df_h = load_data()

with st.spinner("Training AI model..."):
    model, scaler, features, dm = train_model(df, df_h)
# ==========================
# TEST FORECAST
# ==========================

forecast_df = forecast_prices(df, forecast_days=30)

st.subheader("🔮 AI Forecast (Next 30 Days)")
st.dataframe(forecast_df)
# ── CURRENT SIGNAL ────────────────────────────────────────────
latest = dm[features].iloc[-1].values
X_latest = scaler.transform([latest])
proba    = model.predict_proba(X_latest)[0]
signal   = np.argmax(proba)
conf     = proba[signal]
signal_map = {0: ('SELL', 'sell', '🔴'),
              1: ('HOLD', 'hold', '🟡'),
              2: ('BUY',  'buy',  '🟢')}
sig_label, sig_class, sig_icon = signal_map[signal]
current_price = float(dm['Close'].iloc[-1])
# ==========================
# ACCOUNT SUMMARY
# ==========================

portfolio_value = 0

for asset, holding in st.session_state.portfolio.items():
    portfolio_value += holding["shares"] * current_price

total_account_value = st.session_state.cash + portfolio_value
# ── TOP METRICS ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("RSI", f"{dm['rsi'].iloc[-1]:.1f}")
col3.metric("ATR (Volatility)", f"{dm['atr'].iloc[-1]:.2f}")
col4.metric("Confidence", f"{conf:.0%}")
# ==========================
# ACCOUNT SUMMARY
# ==========================

portfolio_value = 0

for asset, holding in st.session_state.portfolio.items():
    portfolio_value += holding["shares"] * current_price

total_account_value = st.session_state.cash + portfolio_value

st.subheader("💰 Account Summary")

a1, a2, a3, a4 = st.columns(4)

a1.metric("Cash", f"${st.session_state.cash:,.2f}")
a2.metric("Portfolio", f"${portfolio_value:,.2f}")
a3.metric("Total Account", f"${total_account_value:,.2f}")
a4.metric("Positions", len(st.session_state.portfolio))
# ==========================
# PAPER TRADING
# ==========================
# Calculate portfolio value
portfolio_value = 0

for asset, holding in st.session_state.portfolio.items():
    portfolio_value += holding["shares"] * current_price
st.subheader("💼 Paper Trading")

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=1,
    step=1
)

col_buy, col_sell = st.columns(2)

with col_buy:
    buy_clicked = st.button("🟢 Buy")

with col_sell:
    sell_clicked = st.button("🔴 Sell")
st.divider()
# ==========================
# PORTFOLIO
# ==========================

st.subheader("📊 Your Portfolio")

if st.session_state.portfolio:

    for asset, holding in st.session_state.portfolio.items():

        current_value = holding["shares"] * current_price
        invested = holding["shares"] * holding["avg_price"]
        profit = current_value - invested

        st.info(
            f"""
**{asset}**

Contracts: **{holding['shares']}**

Average Buy Price: **${holding['avg_price']:.2f}**

Current Value: **${current_value:.2f}**

Profit/Loss: **${profit:.2f}**
"""
        )

else:
    st.warning("No open positions.")
# ==========================
# BUY LOGIC
# ==========================
# ==========================
# TRANSACTION HISTORY
# ==========================

st.subheader("📜 Transaction History")

if st.session_state.transactions:

    for tx in reversed(st.session_state.transactions):

        st.write(
            f"{tx['type']} | "
            f"{tx['quantity']} contract(s) | "
            f"${tx['price']:.2f}"
        )

else:
    st.write("No transactions yet.")
if buy_clicked:
    total_cost = current_price * quantity

    if st.session_state.cash >= total_cost:

        # Deduct cash
        st.session_state.cash -= total_cost

        # Add stock to portfolio
        if "CORN" not in st.session_state.portfolio:
            st.session_state.portfolio["CORN"] = {
                "shares": 0,
                "avg_price": current_price
            }

        holding = st.session_state.portfolio["CORN"]

        total_value = (
            holding["shares"] * holding["avg_price"]
        ) + total_cost

        holding["shares"] += quantity

        holding["avg_price"] = total_value / holding["shares"]

        # Save transaction
        st.session_state.transactions.append({
            "type": "BUY",
            "price": current_price,
            "quantity": quantity
        })

        st.success(f"✅ Bought {quantity} contract(s) at ${current_price:.2f}")

        st.rerun()

    else:
        st.error("❌ Not enough cash.")
        # ==========================
# SELL LOGIC
# ==========================

if sell_clicked:

    if "CORN" in st.session_state.portfolio:

        holding = st.session_state.portfolio["CORN"]

        if holding["shares"] >= quantity:

            sale_value = current_price * quantity

            # Add money back
            st.session_state.cash += sale_value

            # Remove shares
            holding["shares"] -= quantity

            # Save transaction
            st.session_state.transactions.append({
                "type": "SELL",
                "price": current_price,
                "quantity": quantity
            })

            # Remove empty position
            if holding["shares"] == 0:
                del st.session_state.portfolio["CORN"]

            st.success(f"✅ Sold {quantity} contract(s) at ${current_price:.2f}")

            st.rerun()

        else:
            st.error("❌ Not enough contracts to sell.")

    else:
        st.error("❌ You don't own any CORN contracts.")
# ── SIGNAL BOX ────────────────────────────────────────────────
sc1, sc2, sc3 = st.columns([1,2,1])
with sc2:
    st.markdown(f"""
    <div class="signal-box {sig_class}">
        {sig_icon} AI SIGNAL: {sig_label}<br>
        <span style="font-size:18px">Confidence: {conf:.0%}</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.markdown("**Signal Probabilities:**")
    st.progress(float(proba[2]), text=f"BUY  {proba[2]:.0%}")
    st.progress(float(proba[1]), text=f"HOLD {proba[1]:.0%}")
    st.progress(float(proba[0]), text=f"SELL {proba[0]:.0%}")

st.divider()

# ── CHARTS ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Price Chart", "📊 Indicators", "💰 Backtest"
])
with tab1:
    days = st.slider("Days to show", 30, 365, 120)
    d_plot = dm.tail(days)
# ==========================
# AI BUY/SELL PREDICTIONS
# ==========================

X_plot = scaler.transform(d_plot[features])
predictions = model.predict(X_plot)

buy_points = d_plot[predictions == 2]
sell_points = d_plot[predictions == 0]
    # ==========================
    # AI FORECAST
    # ==========================

    forecast_df = forecast_prices(d_plot, forecast_days=30)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Corn Futures Price", "Volume")
    )
    fig.add_trace(
         go.Candlestick(
             x=d_plot.index,
             open=d_plot["Open"],
             high=d_plot["High"],
             low=d_plot["Low"],
             close=d_plot["Close"],
             name="Historical Price"
        ),
        row=1,
        col=1
)
    fig.add_trace(
        go.Candlestick(
            x=forecast_df["Date"],
            open=forecast_df["Open"],
            high=forecast_df["High"],
            low=forecast_df["Low"],
            close=forecast_df["Close"],
            name="🔮 AI Forecast"
       ),
       row=1,
       col=1
)
    for ma, color in [('ma_10','orange'),('ma_20','cyan'),('ma_50','yellow')]:
        fig.add_trace(go.Scatter(
            x=d_plot.index, y=d_plot[ma],
            line=dict(color=color, width=1),
            name=ma.upper()
        ), row=1, col=1)
    colors = ['lime' if s else 'grey' for s in d_plot['volume_spike']]
    fig.add_trace(go.Bar(
        x=d_plot.index, y=d_plot['Volume'],
        marker_color=colors, name='Volume'
    ), row=2, col=1)
    # ==========================
    # AI Forecast Line
    # ==========================

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Close"],
            mode="lines+markers",
            name="🔮 AI Forecast",
            line=dict(
                color="deepskyblue",
                width=4,
                dash="dash"
            ),
            marker=dict(size=7)
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
with tab2:
    d_ind = dm.tail(120)
    fig2 = make_subplots(rows=2, cols=1,
        subplot_titles=("RSI", "MACD"),
        vertical_spacing=0.15)
    fig2.add_trace(go.Scatter(
        x=d_ind.index, y=d_ind['rsi'],
        line=dict(color='purple', width=2), name='RSI'
    ), row=1, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=d_ind.index, y=d_ind['macd'],
        line=dict(color='cyan',   width=1.5), name='MACD'
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=d_ind.index, y=d_ind['macd_signal'],
        line=dict(color='orange', width=1.5), name='Signal'
    ), row=2, col=1)
    fig2.add_trace(go.Bar(
        x=d_ind.index, y=d_ind['macd_hist'],
        marker_color='grey', name='Histogram'
    ), row=2, col=1)
    fig2.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Run backtest
    cash, position, last_trade = 10000, 0, -999
    equity_bt, trades_bt = [], []
    test_data = dm.iloc[-120:]

    for i, (date, row) in enumerate(test_data.iterrows()):
        feat = row[features].values
        X_bt = scaler.transform([feat])
        pb   = model.predict_proba(X_bt)[0]
        sg   = np.argmax(pb)
        cf   = pb[sg]

        if sg == 2 and cf >= 0.45 and position == 0 and (i - last_trade) >= 3:
            position   = (cash * 0.95) / row['Close']
            cash      -= position * row['Close']
            last_trade = i
            trades_bt.append({'Date': str(date)[:10], 'Action': '🟢 BUY',
                             'Price': f"${row['Close']:.2f}", 'Conf': f"{cf:.0%}"})
        elif sg == 0 and cf >= 0.45 and position > 0 and (i - last_trade) >= 3:
            cash      += position * row['Close']
            position   = 0
            last_trade = i
            trades_bt.append({'Date': str(date)[:10], 'Action': '🔴 SELL',
                             'Price': f"${row['Close']:.2f}", 'Conf': f"{cf:.0%}"})

        equity_bt.append(cash + position * row['Close'])

    if position > 0:
        cash += position * test_data['Close'].iloc[-1]

    bh = 10000 * (test_data['Close'].values / test_data['Close'].values[0])
    ret = (equity_bt[-1] - 10000) / 10000 * 100
    bh_ret = (bh[-1] - 10000) / 10000 * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Final Capital",    f"${equity_bt[-1]:,.2f}", f"{ret:+.1f}%")
    m2.metric("Buy & Hold",       f"${bh[-1]:,.2f}",        f"{bh_ret:+.1f}%")
    m3.metric("Total Trades",     str(len(trades_bt)))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(test_data.index), y=equity_bt,
        name='AI Agent', line=dict(color='cyan', width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=list(test_data.index), y=bh,
        name='Buy & Hold', line=dict(color='orange', width=2, dash='dash')
    ))
    fig3.add_hline(y=10000, line_dash="dot", line_color="white", opacity=0.3)
    fig3.update_layout(
        height=400, template='plotly_dark',
        title='Equity Curve — AI Agent vs Buy & Hold'
    )
    st.plotly_chart(fig3, use_container_width=True)

    if trades_bt:
        st.markdown("**Trade History:**")
        st.dataframe(pd.DataFrame(trades_bt), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This tool is for educational purposes only. "
           "Not financial advice. Always do your own research before trading.")
