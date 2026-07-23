import streamlit as st

def show_portfolio(current_price):
    st.title("📊 Portfolio")

    if st.session_state.portfolio:

        total_value = 0

        for asset, holding in st.session_state.portfolio.items():

            current_value = holding["shares"] * current_price
            invested = holding["shares"] * holding["avg_price"]
            profit = current_value - invested

            total_value += current_value

            st.info(f"""
### {asset}

**Contracts:** {holding['shares']}

**Average Buy Price:** ${holding['avg_price']:.2f}

**Current Value:** ${current_value:.2f}

**Profit / Loss:** ${profit:.2f}
""")

        st.divider()

        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

    else:
        st.warning("No open positions.")
