import streamlit as st

def show_forecast(forecast_df):
    st.title("📉 AI Forecast")

    st.markdown("### 🔮 Next 30-Day Forecast")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )
