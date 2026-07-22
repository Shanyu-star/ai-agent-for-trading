import streamlit as st
from database import conn, cursor


def create_user(fullname, email, password):
    try:
        cursor.execute(
            """
            INSERT INTO users(fullname, email, password)
            VALUES (?, ?, ?)
            """,
            (fullname, email, password)
        )
        conn.commit()
        return True
    except:
        return False


def login_user(email, password):
    cursor.execute(
        """
        SELECT * FROM users
        WHERE email=? AND password=?
        """,
        (email, password)
    )

    return cursor.fetchone()


def login_page():

    st.title("🌽 Corn Futures AI Trader")
    st.subheader("Welcome Back")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        login = st.button("Login", use_container_width=True)

    with col2:
        signup = st.button("Create Account", use_container_width=True)

    return login, signup, email, password
