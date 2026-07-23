import streamlit as st

def show_history():
    st.title("📜 Trade History")

    if st.session_state.transactions:

        for tx in reversed(st.session_state.transactions):
            st.write(
                f"{tx['type']} | "
                f"{tx['quantity']} contract(s) | "
                f"${tx['price']:.2f}"
            )

    else:
        st.info("No transactions yet.")
