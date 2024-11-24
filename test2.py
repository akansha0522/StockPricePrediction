import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.express as px
import hashlib
import sqlite3
import time

# Database setup
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True
    return False

# Stock prediction functions 
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def prepare_data(data):
    data['Date'] = data.index.map(pd.Timestamp.toordinal)
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values
    return X, y

def train_model(X, y, model_type, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Decision Tree':
        model = DecisionTreeRegressor(max_depth=params['max_depth'], random_state=0)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=0)

    model.fit(X_train, y_train)
    return model, X_test, y_test

def make_predictions(model, X):
    return model.predict(X)

def main():
    create_table()  # Ensure the database table exists

    st.title("Stock Price Prediction App")

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Select Option", menu)

    if choice == "Register":
        st.subheader("Create a New Account")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Register"):
            if register_user(new_username, new_password):
                st.success("Account created successfully!")
            else:
                st.warning("Username already exists!")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            if check_user(username, password):
                st.session_state['username'] = username
                st.success(f"Welcome {username}!")
            else:
                st.warning("Incorrect username or password!")

    if 'username' in st.session_state:
        st.sidebar.write(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            del st.session_state['username']
            st.success("Logged out successfully.")

        # Proceed with stock prediction features
        ticker = st.text_input("Enter Stock Ticker", "AAPL")  # Default to Apple Inc.
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("today"))
        model_type = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])

        params = {}
        if model_type == 'Decision Tree':
            params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        elif model_type == 'Random Forest':
            params['n_estimators'] = st.slider("Number of Estimators", 10, 200, 100)
            params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        
        data2 = None

        if ticker:
            data = get_stock_data(ticker, start_date, end_date)
            st.subheader(f"Data for {ticker}")
            st.write(data)

            if not data.empty:
                adj_close = data['Adj Close'].values.flatten()
                fig = px.line(x=data.index, y=adj_close, title=f"{ticker} Stock Price")
                st.plotly_chart(fig)

                pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

                with pricing_data:
                    st.header('Price Movements')
                    data2 = data.copy()
                    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
                    data2.dropna(inplace=True)
                    st.write(data2)
                    annual_return = data2['% Change'].mean() * 252 * 100
                    st.write('Annual Return is ', annual_return, '%')
                    stdev = np.std(data2['% Change']) * np.sqrt(252)
                    st.write('Standard Deviation is ', stdev * 100, '%')
                    st.write('Risk Adj. Return is ', annual_return / (stdev * 100))
            else:
                st.warning("No data available for the selected ticker and date range.")

            from stocknews import StockNews
            with news:
                st.header(f'News of {ticker}')
                sn = StockNews(ticker, save_news=False)
                df_news = sn.read_rss()
                for i in range(10):
                    st.subheader(f'News {i+1}')
                    st.write(df_news['published'][i])
                    st.write(df_news['title'][i])
                    st.write(df_news['summary'][i])
                    title_sentiment = df_news['sentiment_title'][i]
                    st.write(f'Title Sentiment {title_sentiment}')
                    news_sentiment = df_news['sentiment_summary'][i]
                    st.write(f'News Sentiment {news_sentiment}')

        if data2 is not None and not data2.empty:
            try:
                X, y = prepare_data(data2)
                model, X_test, y_test = train_model(X, y, model_type, params)

                predictions = make_predictions(model, X_test)

                mae = np.mean(np.abs(predictions - y_test))
                r_squared = model.score(X_test, y_test)

                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"R-squared: {r_squared:.2f}")

                # Create future predictions
                future_days = st.number_input("Days to Predict", min_value=1, max_value=30, value=5)
                future_dates = [data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
                future_dates = future_dates.flatten() if hasattr(future_dates, "flatten") else future_dates
                future_X = np.array([(date.toordinal()) for date in future_dates]).reshape(-1, 1)
                future_predictions = model.predict(future_X)
                future_predictions = future_predictions.flatten() if hasattr(future_predictions, "flatten") else future_predictions

                future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
                st.subheader("Future Predictions")
                st.write(future_df)

                csv = future_df.to_csv(index=False)
                st.download_button("Download Future Predictions", csv, "future_predictions.csv", "text/csv")
            
            except Exception as e:
                st.error(f"An error occurred during modeling: {e}")
        else:
            st.warning("No valid data available for modeling. Please check your inputs.")

if __name__ == "__main__":
    main()
