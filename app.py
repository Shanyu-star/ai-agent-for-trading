# ── LOAD ──────────────────────────────────────────────────────
with st.spinner("Loading corn futures data..."):
    df, df_h = load_data()

with st.spinner("Training AI model..."):
    model, scaler, features, dm = train_model(df, df_h)

# ==========================
# DASHBOARD
# ==========================

if page == "🏠 Dashboard":

    # Forecast
    forecast_df = forecast_prices(df, forecast_days=30)

    st.subheader("🔮 AI Forecast (Next 30 Days)")
    st.dataframe(forecast_df)

    # Current Signal
    latest = dm[features].iloc[-1].values
    X_latest = scaler.transform([latest])
    proba = model.predict_proba(X_latest)[0]

    signal = np.argmax(proba)
    conf = proba[signal]

    signal_map = {
        0: ("SELL", "sell", "🔴"),
        1: ("HOLD", "hold", "🟡"),
        2: ("BUY", "buy", "🟢"),
    }

    sig_label, sig_class, sig_icon = signal_map[signal]

    current_price = float(dm["Close"].iloc[-1])

    # Signal Box
    sc1, sc2, sc3 = st.columns([1, 2, 1])

    with sc2:
        st.markdown(
            f"""
            <div class="signal-box {sig_class}">
                {sig_icon} AI SIGNAL: {sig_label}<br>
                <span style="font-size:18px">
                Confidence: {conf:.0%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Portfolio Value
    portfolio_value = 0

    for asset, holding in st.session_state.portfolio.items():
        portfolio_value += holding["shares"] * current_price

    total_account_value = (
        st.session_state.cash + portfolio_value
    )

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("RSI", f"{dm['rsi'].iloc[-1]:.1f}")
    col3.metric("ATR", f"{dm['atr'].iloc[-1]:.2f}")
    col4.metric("Confidence", f"{conf:.0%}")

    # Account Summary
    st.subheader("💰 Account Summary")

    a1, a2, a3, a4 = st.columns(4)

    a1.metric("Cash", f"${st.session_state.cash:,.2f}")
    a2.metric("Portfolio", f"${portfolio_value:,.2f}")
    a3.metric("Total Account", f"${total_account_value:,.2f}")
    a4.metric("Positions", len(st.session_state.portfolio))
