import praw
import pandas as pd
import re
import string
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- STEP 1: COLLECT DATA FROM REDDIT ---
def fetch_reddit_data(client_id, client_secret, user_agent, subreddit_name='Bitcoin', limit=50000):

    try:
        reddit = praw.Reddit(client_id=client_id,
                             client_secret=client_secret,
                             user_agent=user_agent)
        
        posts_list = []
        subreddit = reddit.subreddit(subreddit_name)
        
        print(f"Collecting {limit} posts from r/{subreddit_name}...")
        for submission in subreddit.top(time_filter='all', limit=None):
            posts_list.append({
                'title': submission.title,
                'created_utc': submission.created_utc
            })
        
        df = pd.DataFrame(posts_list)
        df['created_date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
        print("Collect data successfully.")
        return df
    except Exception as e:
        print(f"Error when collecting data: {e}")
        return None

# --- STEP 2: TEXT PREPROCESSING AND SENTIMENT ANALYSIS ---
def analyze_sentiment(df):

    if df is None:
        return None

    print("\nPreprocessing text and analyzing sentiment...")
    # Initialize analyzer
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))

    # Define a list of Bitcoin price related keywords to filter
    price_keywords = [
        'price', 'buy', 'sell', 'long', 'short', 'bullish', 'bearish',
        'pump', 'dump', 'ath', 'bear', 'bull', 'market', 'trade',
        'investment', 'hodl', 'dip', 'crash', 'rally', 'bitcoin', 'btc', 'coinbase', 'coin'
    ]

    # Preprocessor function
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URL
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
        word_tokens = word_tokenize(text)
        filtered_words = [w for w in word_tokens if not w in stop_words and len(w) > 1]
        
        if not any(word in filtered_words for word in price_keywords):
            return None
        return " ".join(filtered_words)

    # Apply preprocessing and sentiment analysis
    df['processed_title'] = df['title'].apply(preprocess_text)

    # Remove filtered rows
    df.dropna(subset=['processed_title'], inplace=True)
    
    df['sentiment_score'] = df['processed_title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    print("Successful sentiment analysis.")
    return df

# --- STEP 3: AGGREGATE SENTIMENTS OVER TIME ---
def aggregate_daily_sentiment(df):

    if df is None:
        return None
        
    print("\Aggregating average daily sentiment...")
    daily_sentiment_df = df.groupby('created_date')['sentiment_score'].mean().reset_index()
    daily_sentiment_df.columns = ['date', 'avg_sentiment']
    print("Data aggregation successful.")
    return daily_sentiment_df

# --- MAIN ---
if __name__ == "__main__":

    CLIENT_ID = "vsa4ZMTWg_F3YSoIfe61-Q"
    CLIENT_SECRET = "2gRkIrBX92sOUNziZmkjNt13NomR9w"
    USER_AGENT = "appdemo"

    reddit_data = fetch_reddit_data(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    if reddit_data is not None:

        sentiment_data = analyze_sentiment(reddit_data)
        
        daily_sentiment = aggregate_daily_sentiment(sentiment_data)
        
        if daily_sentiment is not None:
            daily_sentiment.to_csv('daily_sentiment.csv', index=False)
            print(f"\nResults saved to file 'daily_sentiment.csv'.")
            print("\nDaily average sentiment summary table:")
            print(daily_sentiment.head())