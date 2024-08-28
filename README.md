Stocket Dashboard 📈
Stocket Dashboard is a comprehensive stock market web application built with Streamlit. It integrates various APIs and tools to provide users with real-time market data, news analysis, technical analysis, and an AI-powered chatbot assistant.

Table of Contents
Features
Technologies Used
Setup and Installation
Function Explanations
Usage
Customization
Contributions
License
Contact
Features
Market Data: Get live updates on stock prices, including various market indicators.
Latest News: Fetch and display news articles related to specific stocks.
Sentiment Analysis: Analyze the sentiment of the news articles and visualize the results.
Technical Analysis: Perform technical analysis on selected stocks using indicators like RSI, MACD, and others.
AI Assistant: Interact with an AI chatbot for stock-related queries and insights.
Technologies Used
Streamlit: Framework for building web applications.
Yahoo Finance (yfinance): Library to fetch historical stock data.
Finnhub API: For live stock market data.
NewsAPI: For fetching news articles.
NLTK (Natural Language Toolkit): For performing sentiment analysis.
Google Gemini AI (Generative AI): For AI-generated insights.
Altair: For data visualization.
Pandas TA: For technical analysis.
MPL Finance: For generating candlestick charts.
Setup and Installation
Prerequisites
Ensure the following are installed:

Python 3.7 or later
Streamlit
Pandas
yfinance
requests
newsapi-python
nltk
google-generativeai
altair
pandas_ta
mplfinance
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/stocket-dashboard.git
cd stocket-dashboard
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Set up API keys:

Add the necessary API keys to the appropriate variables in the code.

Run the application:

bash
Copy code
streamlit run app.py
Function Explanations
1. User Authentication
def authenticate(username, password)
Description: Authenticates the user by checking the provided username and password against a predefined set of credentials.
Parameters: username (str), password (str)
Returns: Boolean value (True if authenticated, False otherwise)
Usage: Validates the login credentials entered by the user.
2. Fetching Live Market Data
def get_live_data(symbol)
Description: Retrieves live market data for the specified stock symbol using the Finnhub API.
Parameters: symbol (str) - The stock ticker symbol.
Returns: Dictionary containing various market metrics (price, open, high, low, etc.)
Usage: Used in the Market Data section to display real-time stock information.
3. Fetching News Articles
def get_news(symbol)
Description: Fetches the latest news articles related to the specified stock symbol using the NewsAPI.
Parameters: symbol (str) - The stock ticker symbol.
Returns: List of dictionaries, each containing news article details (title, description, URL, etc.)
Usage: Displays news articles in the News section of the dashboard.
4. Sentiment Analysis on News Articles
def analyze_sentiment(news_list)
Description: Performs sentiment analysis on a list of news articles to determine the overall sentiment (positive, neutral, or negative).
Parameters: news_list (list) - A list of news articles fetched from get_news().
Returns: A dictionary containing sentiment scores and counts for positive, neutral, and negative articles.
Usage: Provides sentiment insights in the Sentiment Analysis section.
5. Technical Analysis
def perform_technical_analysis(symbol)
Description: Performs technical analysis on the selected stock symbol using various technical indicators (RSI, MACD, OBV, etc.).
Parameters: symbol (str) - The stock ticker symbol.
Returns: A dictionary containing calculated values for each technical indicator.
Usage: Used in the Technical Analysis section to provide advanced market insights.
6. Creating Candlestick Charts
def create_candlestick_chart(symbol)
Description: Generates a candlestick chart for the selected stock symbol using historical data fetched from Yahoo Finance.
Parameters: symbol (str) - The stock ticker symbol.
Returns: None (Directly displays the chart using mplfinance and Streamlit).
Usage: Visualizes historical stock data as a candlestick chart.
7. AI-Powered Chatbot Assistant
def ai_assistant(query)
Description: Uses Google Gemini AI to respond to user queries related to the stock market.
Parameters: query (str) - The user's query.
Returns: A string containing the AI's response.
Usage: Provides insights and answers in the AI Assistant section.
8. Comparing Sentiments Across Stocks
def compare_sentiment(symbols)
Description: Compares sentiment analysis results across multiple stock symbols to provide a comparative view.
Parameters: symbols (list) - A list of stock ticker symbols.
Returns: A dictionary containing sentiment data for each stock symbol.
Usage: Useful for comparing market sentiment across different stocks.
9. Main Application Interface
def main()
Description: The main function that initializes the Streamlit application, manages navigation between different sections, and calls other functions as needed.
Parameters: None
Returns: None
Usage: Acts as the entry point for the application.
Usage
Login: Enter the credentials admin/admin to log in.
Navigate: Use the sidebar to switch between Market Data, News, Sentiment Analysis, Technical Analysis, and AI Assistant.
Explore Data: View live market data, news articles, sentiment analysis, and technical analysis for selected stocks.
Interact: Ask the AI Assistant questions about the stock market for insights.
Customization
Theming: Modify the CSS in the st.markdown() section for custom styling.
Stock Symbols: Add or remove stock symbols by editing the dropdown menu definitions.
Advanced Analysis: Extend the technical analysis functionality by adding more indicators.
