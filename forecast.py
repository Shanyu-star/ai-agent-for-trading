import pandas as pd
import numpy as np
from xgboost import XGBRegressor


def forecast_prices(df, forecast_days=30):
    """
    Forecast future closing prices using XGBoost.
    """

    data = df.copy()

    # Keep required columns
    data = data[["Close"]]

    # ---------- Feature Engineering ----------

    data["lag1"] = data["Close"].shift(1)
    data["lag2"] = data["Close"].shift(2)
    data["lag3"] = data["Close"].shift(3)

    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()

    data = data.dropna()

    features = [
        "lag1",
        "lag2",
        "lag3",
        "MA10",
        "MA20"
    ]

    X = data[features]
    y = data["Close"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    # ----------------------------
    # Recursive Forecast
    # ----------------------------

    history = data.copy()

    predictions = []

    for _ in range(forecast_days):

        latest = history.iloc[-1]

        sample = pd.DataFrame([{
            "lag1": latest["Close"],
            "lag2": history.iloc[-2]["Close"],
            "lag3": history.iloc[-3]["Close"],
            "MA10": history["Close"].tail(10).mean(),
            "MA20": history["Close"].tail(20).mean()
        }])

        pred = model.predict(sample)[0]

        predictions.append(pred)

        new_row = latest.copy()
        new_row["Close"] = pred

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predictions
    })

    return forecast_df
