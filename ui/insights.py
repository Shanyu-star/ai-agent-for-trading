import streamlit as st

def show_insights(sig_icon, sig_label, conf, proba):

    st.title("🤖 AI Insights")

    st.markdown(f"""
    <div class="signal-box">
        <h2>{sig_icon} AI SIGNAL : {sig_label}</h2>
        <h3>Confidence : {conf:.0%}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Signal Probability")

    st.progress(float(proba[2]), text=f"BUY {proba[2]:.0%}")
    st.progress(float(proba[1]), text=f"HOLD {proba[1]:.0%}")
    st.progress(float(proba[0]), text=f"SELL {proba[0]:.0%}")
