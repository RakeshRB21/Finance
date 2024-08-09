import streamlit as st
import yfinance as yf
import altair as alt
import pandas as pd
import requests
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# API keys
FINNHUB_API_KEY = "cqpsqd9r01qo7iavso20cqpsqd9r01qo7iavso2g"
NEWS_API_KEY = "773f803ad36b4f298c176361f14741bd"
GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to check username and password
def check_credentials():
    """Returns `True` if the user had the correct username and password."""
    def credentials_entered():
        """Checks whether the username and password entered by the user are correct."""
        if (st.session_state["username"] == "admin" and
                st.session_state["password"] == "admin"):
            st.session_state["credentials_correct"] = True
            del st.session_state["username"]  # Don't store the username
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["credentials_correct"] = False

    if "credentials_correct" not in st.session_state:
        # First run, show input for username and password.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=credentials_entered)
        return False
    elif not st.session_state["credentials_correct"]:
        # Credentials not correct, show input and error.
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=credentials_entered)
        st.error("😕 Username or password incorrect")
        return False
    else:
        # Credentials correct.
        return True

# Function to fetch live stock data from Finnhub
def fetch_live_stock_data(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to fetch news articles using NewsAPI
def fetch_news(symbol):
    news_api = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = news_api.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=10)
    return all_articles['articles']

# Function to perform sentiment analysis on news articles
def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles:
        sentiment = analyzer.polarity_scores(article['content'])
        sentiments.append(sentiment)
    return sentiments

# Function to get chatbot response using Groq AI
def get_chatbot_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            },
        ],
        model="gemma-7b-it",
    )
    return chat_completion.choices[0].message.content

# Function to build and train a predictive model
def build_predictive_model(stock_data):
    # Feature engineering
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data = stock_data.dropna()

    # Prepare data for training
    X = stock_data[['MA50', 'MA200']]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    return model, X_test, y_test, y_pred

# Function to simulate real-time streaming using Altair
def real_time_streaming(symbol):
    st.subheader("Real-Time Stock Data Streaming")

    # Container to hold the chart
    chart_placeholder = st.empty()

    # Fetch initial stock data using yfinance
    stock_data = yf.download(symbol, period='1d', interval='1m')
    stock_data.reset_index(inplace=True)

    # Create an Altair line chart
    base_chart = alt.Chart(stock_data).mark_line().encode(
        x='Datetime:T',
        y='Close:Q'
    ).properties(
        width=700,
        height=400
    )

    # Display the initial chart
    chart_placeholder.altair_chart(base_chart)

    # Stream data every minute
    while True:
        # Fetch new stock data
        new_data = yf.download(symbol, period='1d', interval='1m')
        new_data.reset_index(inplace=True)

        # Update the chart with new data
        updated_chart = alt.Chart(new_data).mark_line().encode(
            x='Datetime:T',
            y='Close:Q'
        ).properties(
            width=700,
            height=400
        )

        # Update the Streamlit chart
        chart_placeholder.altair_chart(updated_chart)

        # Sleep for a short duration before fetching new data
        time.sleep(60)

# Main function
def main():
    if check_credentials():
        st.title("Stock Squawk Dashboard")

        # Sidebar for user input
        st.sidebar.header("User Input")
        stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")

        # Fetch and display live stock data
        if st.sidebar.button("Fetch Stock Data", key="fetch_stock_data_button"):
            stock_data = fetch_live_stock_data(stock_symbol)
            if stock_data:
                st.subheader("Live Stock Data")
                df = pd.DataFrame.from_dict(stock_data, orient='index', columns=["Value"])
                df.index = ["Current Price", "High", "Low", "Open", "Previous Close", "Timestamp", "Change", "Percent Change"]
                st.table(df)
            else:
                st.error("Failed to fetch stock data.")

        # Fetch and display news articles
        if st.sidebar.button("Fetch News", key="fetch_news_button"):
            articles = fetch_news(stock_symbol)
            if articles:
                st.subheader("Top 10 News Articles")
                news_data = []
                for article in articles:
                    news_data.append({
                        "Title": article['title'],
                        "Description": article['description'],
                        "URL": article['url']
                    })
                news_df = pd.DataFrame(news_data)
                st.table(news_df.head(10))
            else:
                st.error("Failed to fetch news articles.")

        # Perform sentiment analysis
        if st.sidebar.button("Analyze Sentiment", key="analyze_sentiment_button"):
            articles = fetch_news(stock_symbol)
            if articles:
                sentiments = analyze_sentiment(articles)
                st.subheader("Sentiment Analysis of Top 10 News Articles")
                sentiment_data = []
                for idx, sentiment in enumerate(sentiments):
                    sentiment_data.append({
                        "Article": articles[idx]['title'],
                        "Positive": sentiment['pos'],
                        "Neutral": sentiment['neu'],
                        "Negative": sentiment['neg'],
                        "Compound": sentiment['compound']
                    })
                sentiment_df = pd.DataFrame(sentiment_data)
                st.table(sentiment_df.head(10))
            else:
                st.error("Failed to perform sentiment analysis.")

        # Chatbot interaction
        st.sidebar.subheader("Chatbot Interaction")
        user_input = st.sidebar.text_input("Ask a question", key="chatbot_input")
        if st.sidebar.button("Get Response", key="chatbot_response_button"):
            response = get_chatbot_response(user_input)
            st.subheader("Chatbot Response")
            st.write(response)

        # Real-time stock data streaming
        if st.sidebar.button("Start Real-Time Streaming", key="real_time_streaming_button"):
            real_time_streaming(stock_symbol)

        # Train and display predictive model results
        if st.sidebar.button("Train Predictive Model", key="train_predictive_model_button"):
            stock_data = yf.download(stock_symbol, period='1y', interval='1d')
            model, X_test, y_test, y_pred = build_predictive_model(stock_data)
            st.subheader("Predictive Model Results")
            st.write("Actual vs Predicted:")
            result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.table(result_df.head(10))

# Run the application
if __name__ == "__main__":
    main()