import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import google.generativeai as genai
import altair as alt
import pandas_ta as ta
import mplfinance as mpf
from datetime import datetime


#1. Initial Setup
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

#2. API keys
FINNHUB_API_KEY = "cqpsqd9r01qo7iavso20cqpsqd9r01qo7iavso2g"
NEWS_API_KEY = "773f803ad36b4f298c176361f14741bd"
GEMINI_API_KEY = "AIzaSyCBwz4HwTuOh4NMEU0BSmFcFSwImWXoE8E"

#3. Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

#4. Custom Theme and Styling
st.set_page_config(page_title="Stocket Dashboard", page_icon="📈", layout="wide")

#5. Custom CSS for improved aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    .css-1d391kg {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #FF69B4;
        color: white;
        border-radius: 50px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF1493;
    }
    .stSelectbox>div>div {
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #FF69B4;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 1rem;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a {
        color: #000000;
        text-decoration: none;
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a:hover {
        background-color: #FF69B4;
    }
    .sidebar .block-container div[data-testid="stSidebarNav"] a svg {
        margin-right: 0.5rem;
    }
    .news-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    .news-item {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        flex: 1 1 calc(33.333% - 20px);
    }
    .news-item:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .news-item h3 {
        margin-bottom: 10px;
        color: #FF69B4;
    }
    .news-item p {
        margin-bottom: 10px;
        color: #333333;
    }
    .news-item a {
        color: #FF69B4;
        text-decoration: none;
    }
    .news-item a:hover {
        text-decoration: underline;
    }
    .news-item .news-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .news-item .news-meta span {
        transition: color 0.3s ease, text-shadow 0.3s ease;
    }
    .news-item .news-meta span:hover {
        color: #FF69B4;
        text-shadow: 0 0 5px #FF69B4;
    }
    .heatmap-container {
        margin-top: 20px;
    }
    .comparison-container {
        margin-top: 20px;
    }
    .explanation-container {
        margin-top: 20px;
    }
    .sentiment-explanation {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sentiment-explanation h4 {
        color: #FF69B4;
    }
    .sentiment-explanation p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

#6. Authentication Function
def check_credentials():
    if "credentials_correct" not in st.session_state:
        st.title('Welcome to Stocket Dashboard')
        st.subheader('Please login to continue')
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state["credentials_correct"] = True
                st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
            else:
                st.error("😕 Username or password incorrect")
        return False
    return st.session_state["credentials_correct"]

#7. Data Fetching Functions
@st.cache_data(ttl=300)
def get_live_market_data(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

@st.cache_data(ttl=3600)
def get_historical_data(symbol, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return yf.download(symbol, start=start_date, end=end_date)

#8. News and Sentiment Analysis Functions
@st.cache_data(ttl=3600)
def fetch_news(symbol):
    news_api = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = news_api.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=10)
    return all_articles['articles']

def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(article['content']) for article in articles]

#9. Technical Analysis Functions
def perform_technical_analysis(data):
    if data is None or data.empty:
        st.error("No data available for technical analysis.")
        return None

    try:
        # RSI
        data['RSI'] = ta.rsi(data['Close'], length=14)

        # MACD
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_Signal'] = macd['MACDs_12_26_9']
        data['MACD_Diff'] = macd['MACDh_12_26_9']

        # OBV
        data['OBV'] = ta.obv(data['Close'], data['Volume'])

        # ADL
        data['ADL'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
        data['ADL'] = data['ADL'].cumsum()

        # ADX
        adx = ta.adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx['ADX_14']

        # Stochastic
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        data['Stochastic_K'] = stoch['STOCHk_14_3_3']
        data['Stochastic_D'] = stoch['STOCHd_14_3_3']

        return data
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        return None

def technical_analysis_page():
    st.title("🧠 AI-Powered Technical Analysis")
    stock_symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    data = get_historical_data(stock_symbol, days=365)

    if not data.empty:
        data = perform_technical_analysis(data)

        st.subheader("Technical Indicators")
        indicator_options = ["RSI", "MACD", "MACD_Signal", "MACD_Diff", "OBV", "ADL", "ADX", "Stochastic_K", "Stochastic_D"]
        selected_indicators = st.multiselect("Select Indicators", indicator_options, default=["RSI", "MACD"])

        if selected_indicators:
            st.line_chart(data[selected_indicators])
        else:
            st.warning("Please select at least one indicator to display.")

        # AI-powered trend analysis
        trend_prompt = f"Analyze the following technical indicators for {stock_symbol}:\n{data[indicator_options].tail().to_string()}\nProvide a concise analysis of the current trend and potential future movement."
        trend_response = model.generate_content(trend_prompt)
        st.write(trend_response.text)
        
#10. Page Functions
def market_data_page():
    st.title("📊 Market Data")
    stock_symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])

    stock_data = get_live_market_data(stock_symbol)
    if stock_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Current Price", value=f"${stock_data['c']:.2f}", delta=f"{stock_data['c'] - stock_data['pc']:.2f}")
        with col2:
            st.metric(label="Open", value=f"${stock_data['o']:.2f}")
        with col3:
            st.metric(label="High", value=f"${stock_data['h']:.2f}")
        with col4:
            st.metric(label="Low", value=f"${stock_data['l']:.2f}")

        historical_data = get_historical_data(stock_symbol)
        historical_data.reset_index(inplace=True)
        historical_data = historical_data.melt(id_vars=['Date'], value_vars=['Open', 'High', 'Low', 'Close'], var_name='Type', value_name='Price')

        chart = alt.Chart(historical_data).mark_area(
            opacity=0.3,
            interpolate='basis'
        ).encode(
            x='Date:T',
            y='Price:Q',
            color='Type:N',
            tooltip=['Date', 'Type', 'Price']
        ).properties(
            title=f"{stock_symbol} Stacked Area Chart (Last 30 Days)",
            width=800,
            height=400
        ).configure_mark(
            opacity=0.7
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

def news_page():
    st.title("📰 Latest Stock News")
    stock_symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])

    sort_option = st.selectbox("Sort by", ["Relevance", "Date"])

    articles = fetch_news(stock_symbol)
    if articles:
        if sort_option == "Date":
            articles = sorted(articles, key=lambda x: x['publishedAt'], reverse=True)

        st.markdown("""
        <style>
        .news-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            padding: 2rem;
        }
        .news-card {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .news-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        }
        .news-content {
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        .news-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            line-height: 1.4;
        }
        .news-description {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 1.5rem;
            flex-grow: 1;
            line-height: 1.6;
        }
        .news-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: #999;
            margin-top: auto;
        }
        .read-more {
            background: #FF69B4;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-decoration: none;
            transition: background 0.3s ease;
            font-weight: bold;
        }
        .read-more:hover {
            background: #FF1493;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="news-container">', unsafe_allow_html=True)
        for article in articles:
            st.markdown(f'''
            <div class="news-card">
                <div class="news-content">
                    <h3 class="news-title">{article['title']}</h3>
                    <p class="news-description">{article['description']}</p>
                    <div class="news-meta">
                        <span>{format_published_date(article['publishedAt'])}</span>
                        <a href="{article['url']}" target="_blank" class="read-more">Read More</a>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No news articles found for the selected stock symbol.")
        
#11. Helper functions for the news_page()
def format_published_date(published_date):
    # Parse the ISO 8601 formatted date string
    parsed_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
    # Format the date to a more user-friendly format
    return parsed_date.strftime("%b %d, %Y")

def sentiment_analysis_page():
    st.title("🔍 Sentiment Analysis")
    stock_symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    
    articles = fetch_news(stock_symbol)
    if articles:
        sentiments = analyze_sentiment(articles)
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame({
            'Title': [article['title'] for article in articles],
            'Positive': [sentiment['pos'] for sentiment in sentiments],
            'Neutral': [sentiment['neu'] for sentiment in sentiments],
            'Negative': [sentiment['neg'] for sentiment in sentiments],
            'Compound': [sentiment['compound'] for sentiment in sentiments]
        })
        
        st.subheader("Sentiment Heatmap")
        
        # Create a more defined heatmap using Altair
        heatmap = alt.Chart(df.melt(id_vars=['Title'], var_name='Sentiment', value_name='Score')).mark_rect().encode(
            x=alt.X('Sentiment:N', title=None),
            y=alt.Y('Title:N', title=None, sort='-x'),
            color=alt.Color('Score:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Title', 'Sentiment', 'Score']
        ).properties(
            width=600,
            height=400,
            title=f"Sentiment Analysis Heatmap for {stock_symbol}"
        )
        
        st.altair_chart(heatmap, use_container_width=True)
        
        st.subheader("Detailed Sentiment Scores")
        st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Positive', 'Neutral', 'Negative', 'Compound']))
        
        # Add an overall sentiment summary
        avg_sentiment = df['Compound'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        st.subheader("Overall Sentiment Summary")
        st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", sentiment_label)
        
        # Generate AI insights
        prompt = f"Analyze the following sentiment scores for {stock_symbol} news articles:\n{df.to_string()}\nProvide a brief, insightful summary of the overall sentiment and its potential impact on the stock."
        response = model.generate_content(prompt)
        st.subheader("AI Insights")
        st.write(response.text)

        # Sentiment Score Explanation
        st.markdown('<div class="explanation-container">', unsafe_allow_html=True)
        st.subheader("Sentiment Score Explanation")
        for i, article in enumerate(articles[:5]):
            with st.expander(f"**{article['title']}**"):
                st.write(f"Positive: {sentiments[i]['pos']}, Neutral: {sentiments[i]['neu']}, Negative: {sentiments[i]['neg']}, Compound: {sentiments[i]['compound']}")
                st.write(f"Key Phrases: {', '.join([word for word in article['content'].split() if word.lower() not in nltk.corpus.stopwords.words('english')])}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Sentiment Comparison
        st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
        st.subheader("Sentiment Comparison")
        comparison_symbols = st.multiselect("Select Stocks to Compare", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], default=[stock_symbol])
        comparison_data = {}
        for symbol in comparison_symbols:
            comparison_articles = fetch_news(symbol)
            comparison_sentiments = analyze_sentiment(comparison_articles)
            comparison_data[symbol] = pd.DataFrame({
                'Article': [a['title'] for a in comparison_articles],
                'Compound': [s['compound'] for s in comparison_sentiments]
            })
        comparison_df = pd.concat(comparison_data.values(), keys=comparison_data.keys())
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'level_0': 'Symbol'}, inplace=True)

        comparison_chart = alt.Chart(comparison_df).mark_boxplot(extent='min-max').encode(
            x='Symbol:N',
            y='Compound:Q',
            color='Symbol:N',
            tooltip=['Symbol', 'Compound']
        ).properties(
            title='Sentiment Comparison',
            width=800,
            height=400
        ).interactive()

        st.altair_chart(comparison_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def chatbot_page():
    st.title("🤖 Stocket AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about stocks?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in model.generate_content(prompt, stream=True):
                full_response += response.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#12. Main Dashboard Function
def main():
    if check_credentials():
        st.sidebar.title("Navigation")

        if st.sidebar.button("📊 Market Data"):
            st.session_state.page = "market_data"
        if st.sidebar.button("📰 News"):
            st.session_state.page = "news"
        if st.sidebar.button("🔍 Sentiment Analysis"):
            st.session_state.page = "sentiment_analysis"
        if st.sidebar.button("📊 AI-Powered Technical Analysis"):
            st.session_state.page = "technical_analysis"
        if st.sidebar.button("🤖 AI Assistant"):
            st.session_state.page = "ai_assistant"

        if "page" not in st.session_state:
            st.session_state.page = "market_data"

        if st.session_state.page == "market_data":
            market_data_page()
        elif st.session_state.page == "news":
            news_page()
        elif st.session_state.page == "sentiment_analysis":
            sentiment_analysis_page()
        elif st.session_state.page == "technical_analysis":
            technical_analysis_page()
        elif st.session_state.page == "ai_assistant":
            chatbot_page()

if __name__ == "__main__":
    main()
