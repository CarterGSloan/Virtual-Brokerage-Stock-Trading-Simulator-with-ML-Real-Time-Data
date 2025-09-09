import math
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class StockPredictor:
    """
    Trains a decision tree model on historical stock data to predict next-day movement (UP or DOWN).
    Provides a prediction and a basic reasoning based on recent trends.
    """
    def __init__(self, history_df):
        """
        Initialize the predictor with historical data.
        history_df: pandas DataFrame containing columns 'Open', 'Close', 'Volume' for each date.
        """
        self.history = history_df
        self.model = None
        self.last_features = None
        if history_df is None or len(history_df) < 2:
            #Not enough data to train (need at least 2 days)
            return
        #Prepare training data
        X = []
        y = []
        #Use all days except the last one for training
        for i in range(len(history_df) - 1):
            today_open = history_df['Open'].iloc[i]
            today_close = history_df['Close'].iloc[i]
            today_vol = history_df['Volume'].iloc[i]
            tomorrow_close = history_df['Close'].iloc[i+1]
            if math.isnan(today_open) or math.isnan(today_close) or math.isnan(tomorrow_close):
                continue    # skip days with missing data
            #Feature: percentage change during the day
            change_pct = ((today_close - today_open) / today_open * 100.0) if today_open != 0 else 0.0
            #Label: whether next day went up or down
            label = "UP" if tomorrow_close > today_close else "DOWN"
            X.append([today_open, today_close, today_vol, change_pct])
            y.append(label)
        if len(X) == 0:
            return  # no valid training instances
        X = np.array(X)
        #Train a Decision Tree Classifier
        self.model = DecisionTreeClassifier(random_state=0)
        self.model.fit(X, y)
        #Store features from the last avaliable day for prediction
        last_idx = len(history_df) - 1
        last_open = history_df['Open'].iloc[last_idx]
        last_close = history_df['Close'].iloc[last_idx]
        last_vol = history_df['Volume'].iloc[last_idx]
        if not math.isnan(last_open) and not math.isnan(last_close):
            last_change_pct = ((last_close - last_open) / last_open * 100.0) if last_open != 0 else 0.0
            self.last_features = [last_open, last_close, last_vol, last_change_pct]

    def predict_next_day(self):
        """
        Predict whether the stock will go UP or DOWN on the next trading day.
        Returns a tuple: (prediction_str, reasoning_str).
        """
        if self.model is None or self.last_features is None:
            return ("UKNOWN", "Not enough data to predict.")
        #Classify using the training model
        pred_label = self.model.predict([self.last_features])[0]
        #Analyze recent trend for reasoning
        closes = self.history['Close']
        #Consider last 5 days (or fewer if history is shorter)
        recent_prices = list(closes.iloc[-5:]) if len(closes) >= 5 else list(closes)
        ups = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        downs = (len(recent_prices) - 1) - ups
        trend = "mixed"
        if len(recent_prices) > 1:
            if ups >= 2 and ups >= downs:
                trend = "up"
            elif downs >= 2 and downs > ups:
                trend = "down"
        #Formulate a simple explanation based on trend vs prediction
        if pred_label == "UP":
            if trend == "up":
                reasoning = "The stock has upward momentum recently, which may continue."
            elif trend == "down":
                reasoning = "Despite a recent downward trend, the model anticipates a rebound."
            else:
                reasoning = "The model predicts an upward movement for tomorrow."
        else:  # pred_label == "DOWN"
            if trend == "down":
                reasoning = "The stock has been on a downward trend, which may persist."
            elif trend == "up":
                reasoning = "Despite recent gains, the model indicates a potential downturn."
            else:
                reasoning = "The model predicts a downward movement for tomorrow."
        return (pred_label, reasoning)