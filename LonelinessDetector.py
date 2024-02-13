import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Import data sets
r_lonely_data = pd.read_csv('r_lonely.csv')
r_casualConversation_data = pd.read_csv('r_casualConversation.csv')

# measure_sentiment_intensity that measures the sentiment intensity of a sentence
def measure_sentiment_intensity(text_data):
    return sia.polarity_scores(text_data)

# average_sentiment, which calculate the average sentiment score by averaging both the title's sentiment score and the post's
# sentiment score from a data in the dataframe.
def average_sentiment(title, post):
    title_scores = measure_sentiment_intensity(title)
    post_scores = measure_sentiment_intensity(post)
    average_neg = (title_scores['neg'] + post_scores['neg']) / 2
    average_neu = (title_scores['neu'] + post_scores['neu']) / 2
    average_pos = (title_scores['pos'] + post_scores['pos']) / 2

    return average_neg, average_neu, average_pos

# Create a new list to store the result and convert to dataframe later.
sentiment_results = []

# Iterate through the dataframe r_lonely_data, and get the sentiment score for each row.
for index, row in r_lonely_data.iterrows():
    avg_neg, avg_neu, avg_pos = average_sentiment(row['Title'], row['Post'])
    sentiment_results.append({'Average_Neg' : avg_neg, 'Average_Neu' : avg_neu, 'Average_Pos' : avg_pos})

# Create the dataframe using the results above.
sentiment_df = pd.DataFrame(sentiment_results)

# Playing with the data
print(sentiment_df.shape)
print(sentiment_df.head(10))
print(r_lonely_data.iloc[8])