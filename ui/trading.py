import streamlit as st

def show_trading(current_price):
    st.title("📈 Paper Trading")

    st.info(f"Current Corn Price: ${current_price:.2f}")

    quantity = st.number_input(
        "Contracts",
        min_value=1,
        value=1,
        step=1
    )

    col1, col2 = st.columns(2)

    buy = col1.button("🟢 Buy")
    sell = col2.button("🔴 Sell")

    # ---------------- BUY ----------------
    if buy:
        total_cost = current_price * quantity

        if st.session_state.cash >= total_cost:

            st.session_state.cash -= total_cost

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

            holding["avg_price"] = (
                total_value / holding["shares"]
            )

            st.session_state.transactions.append({
                "type": "BUY",
                "price": current_price,
                "quantity": quantity
            })

            st.success(
                f"Bought {quantity} contract(s)"
            )

            st.rerun()

        else:
            st.error("Not enough cash.")

    # ---------------- SELL ----------------
    if sell:

        if "CORN" not in st.session_state.portfolio:
            st.error("No CORN contracts.")
            return

        holding = st.session_state.portfolio["CORN"]

        if holding["shares"] < quantity:
            st.error("Not enough contracts.")
            return

        sale_value = current_price * quantity

        st.session_state.cash += sale_value

        holding["shares"] -= quantity

        st.session_state.transactions.append({
            "type": "SELL",
            "price": current_price,
            "quantity": quantity
        })

        if holding["shares"] == 0:
            del st.session_state.portfolio["CORN"]

        st.success(
            f"Sold {quantity} contract(s)"
        )

        st.rerun()
