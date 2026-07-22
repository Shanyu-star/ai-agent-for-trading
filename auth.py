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
