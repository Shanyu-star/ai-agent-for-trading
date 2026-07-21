import pandas as pd
import numpy as np
from xgboost import XGBRegressor
np.random.seed(42)

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
    # Improved Recursive Forecast
    # ----------------------------

    history = data.copy()
    predictions = []

    recent_volatility = history["Close"].pct_change().std()

    trend = (
        history["Close"].tail(10).mean()
        - history["Close"].tail(30).mean()
    ) / 30

    last_price = history.iloc[-1]["Close"]

    for day in range(forecast_days):

        sample = pd.DataFrame([{
            "lag1": history.iloc[-1]["Close"],
            "lag2": history.iloc[-2]["Close"],
            "lag3": history.iloc[-3]["Close"],
            "MA10": history["Close"].tail(10).mean(),
            "MA20": history["Close"].tail(20).mean()
        }])

        base_prediction = model.predict(sample)[0]

        noise = np.random.normal(
            0,
            recent_volatility * last_price * 0.5
        )

        predicted = (
            base_prediction
            + trend * day
            + noise
        )

        predictions.append(predicted)

        new_row = history.iloc[-1].copy()
        new_row["Close"] = predicted

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

        last_price = predicted
            future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    future_open = []
    future_high = []
    future_low = []
    future_close = []
    future_volume = []

    prev_close = data["Close"].iloc[-1]

    for close in predictions:

        open_price = prev_close

        spread = abs(close - open_price)

        high = max(open_price, close) + np.random.uniform(
            0.3, 1.0
        ) * (spread + 2)

        low = min(open_price, close) - np.random.uniform(
            0.3, 1.0
        ) * (spread + 2)

        volume = np.random.randint(
            int(df["Volume"].tail(20).mean() * 0.8),
            int(df["Volume"].tail(20).mean() * 1.2)
        )

        future_open.append(open_price)
        future_high.append(high)
        future_low.append(low)
        future_close.append(close)
        future_volume.append(volume)

        prev_close = close

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Open": future_open,
        "High": future_high,
        "Low": future_low,
        "Close": future_close,
        "Volume": future_volume
    })

    return forecast_df

