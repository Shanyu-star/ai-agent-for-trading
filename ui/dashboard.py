import streamlit as st

def show_dashboard(dm, current_price, conf, sig_label, sig_icon, portfolio_value):
    """Dashboard Page"""

    st.title("🏠 Dashboard")

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("RSI", f"{dm['rsi'].iloc[-1]:.1f}")
    col3.metric("ATR", f"{dm['atr'].iloc[-1]:.2f}")
    col4.metric("Confidence", f"{conf:.0%}")

    st.divider()

    st.subheader("🤖 AI Trading Signal")

    if sig_label == "BUY":
        st.success(f"{sig_icon} Strong BUY Signal ({conf:.0%})")

    elif sig_label == "SELL":
        st.error(f"{sig_icon} Strong SELL Signal ({conf:.0%})")

    else:
        st.warning(f"{sig_icon} HOLD ({conf:.0%})")

    st.divider()

    st.subheader("💰 Account Summary")

    total = st.session_state.cash + portfolio_value

    a1, a2, a3 = st.columns(3)

    a1.metric("Cash", f"${st.session_state.cash:,.2f}")
    a2.metric("Portfolio", f"${portfolio_value:,.2f}")
    a3.metric("Total Value", f"${total:,.2f}")
