"""
This module provides functions for analyzing sentiment in YouTube comments
using training an SVM model
"""
# Import required packages
import json
import re
import urllib.parse
import urllib.request

import googleapiclient.discovery
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from langdetect import detect
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
matplotlib.use('Agg')


def extract_yt_comments(video_id, youtube_api_key):
    """
    Extract YouTube comments using YouTube data API
    arguments:
        video_id: YouTube video id str type
        youtube_api_key: static API key for accessing the YouTube Data API to fetch comments
    returns:
        dataframe: dataframe containing extracted comments
    """
    print("Extracting comments for video id:", video_id)
    youtube = googleapiclient.discovery.build("youtube", "v3",
                                              developerKey=youtube_api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()
    comments = []

    # Read comments in loop until next_page_token exist(read all comments from the video)
    while True:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['textDisplay'],
                comment['publishedAt'],
                comment['likeCount']
            ])

        try:
            next_page_token = response['nextPageToken']
        except KeyError:
            break

        next_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )

        response = next_request.execute()
    # Return a df containing all the extracted comments
    return pd.DataFrame(comments, columns=['Author_Name', 'Comment_Text',
                                           'Updated_Time', 'Likes_Count'])


def clean_comments(comments_df):
    """
    Clean the raw extracted comments
    arguments:
        comments_df: dataframe containing comments
    returns:
        text: cleaned dataframe with comments
    """
    print("Clean comments")

    # Reference:https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    def remove_emoji_comments(comment):
        """Remove emojis, symbols, flags from all comments"""

        if not isinstance(comment, str):
            comment = str(comment)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        cleaned_comment = comment.lower()
        cleaned_comment = emoji_pattern.sub(r'', cleaned_comment)
        cleaned_comment = re.sub(r'\d+', '', cleaned_comment)
        cleaned_comment = cleaned_comment.strip()
        return cleaned_comment

    comments_df.dropna(inplace=True)
    comments_df = comments_df.drop_duplicates()
    comments_df.drop(columns=['Likes_Count', 'Author_Name', 'Updated_Time'], inplace=True)
    comments_df["Comment_Text"] = comments_df["Comment_Text"].apply(remove_emoji_comments)
    comments_df["Comment_Text"] = comments_df["Comment_Text"].str.replace("\n", "", regex=False)
    comments_df["Comment_Text"] = comments_df["Comment_Text"].str.replace(",", " ", regex=False)
    comments_df["Comment_Text"] = comments_df["Comment_Text"].str.replace("?", " ", regex=False)
    comments_df["Comment_Text"] = comments_df["Comment_Text"].str.replace("!", " ", regex=False)

    # Remove any comment if it's empty after data cleaning
    zero_length_comments = comments_df[comments_df["Comment_Text"].map(len) == 0]
    comments_df = comments_df.drop(zero_length_comments.index)
    comments_df.reset_index(drop=True, inplace=True)

    return comments_df


def remove_consecutive_spaces(text):
    """
    Remove multiple consecutive spaces in a comment
    arguments:
        text: string comment
    returns:
        text: comment after removing more than single spaces
    """
    return re.sub(r'\s{2,}', ' ', text)


def detect_and_drop_non_english(df):
    """
    Check language for each comment and drop any non-english comment
    arguments:
        df: dataframe containing comments
    returns:
        df: dataframe containing comments after removing non english comments
    """
    print("Detecting non english comments")
    non_english_indices = []
    for i, text in enumerate(df['Comment_Text']):
        try:
            if detect(text) != 'en':
                non_english_indices.append(i)
        except:
            continue
    df = df.drop(non_english_indices)
    df.reset_index(drop=True, inplace=True)
    print("detect_and_drop_non_english DONE")
    return df


def tokenize_comment(comment):
    """
    Tokenizes and lemmatizes the comment.
    arguments:
        comment: str type comment to be tokenized and lemmatized
    returns:
        str: tokenized and lemmatized comment
    """
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    word_tokens = word_tokenize(comment)
    filtered_text = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_text.append(w)
    text = ' '.join([wnl.lemmatize(word) for word in filtered_text])
    return text


def evaluate_polarity_comments(comment):
    """
    Evaluate the polarity of given comment using vader sentiment analysis.
    arguments:
        comment: str type comment to evaluate polarity
    returns:
        tuple: positive score, negative score, neutral score,
               compound score, and sentiment label of the comment.
    """
    score = SentimentIntensityAnalyzer().polarity_scores(comment)
    if score['compound'] >= 0.05:
        sentiment = "Positive"
    elif score['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return score['pos'], score['neg'], score['neu'], score['compound'], sentiment


def analyze_sentiment(comments_df):
    """
    Analyze the sentiment of comments in a DataFrame
    arguments:
        comments_df: tokenized comments dataframe
    returns:
        tuple: tuple containing the labelled comments, positive, negative and neutral count
    """
    # Apply the evaluate_polarity_comments function to each tokenized comment
    polarized_comments_data = []
    for comment in comments_df['tokenized_comment']:
        pos, neg, neu, compound, sentiment = evaluate_polarity_comments(comment)
        polarized_comments_data.append([comment, pos, neg, neu, compound, sentiment])

    # Create a DataFrame from the polarized comments data
    labelled_comments_data = pd.DataFrame(polarized_comments_data,
                                          columns=['Comment', 'Positive',
                                                   'Negative', 'Neutral',
                                                   'Compound', 'Sentiment'])

    # Count the occurrences of each sentiment
    sentiment_counts = labelled_comments_data['Sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)

    # Print the counts of each sentiment
    print("Positive Count:", positive_count)
    print("Negative Count:", negative_count)
    print("Neutral Count:", neutral_count)
    print(labelled_comments_data.head(10))

    # Return the labelled comments Data Frame
    return labelled_comments_data, positive_count, negative_count, neutral_count

def preprocess_comments(comments_df):
    """
    Preprocess the comments
    arguments:
        comments_df: comments dataframe
    returns:
        DataFrame: preprocessed tokenized comments
    """
    print("Preprocessing comments")
    comments_df['Comment_Text'] = comments_df['Comment_Text'].apply(remove_consecutive_spaces)
    comments_df = detect_and_drop_non_english(comments_df)
    comments_df['tokenized_comment'] = comments_df["Comment_Text"].apply(tokenize_comment)
    return comments_df


def fetch_youtube_title(video_id):
    """
    Fetch YouTube video title using its video id
    arguments:
        video_id: YouTube video id str type
    returns:
        str: title of the YouTube video
    """
    # Reference: https://stackoverflow.com/questions/30084140/youtube-video-title-with-api-v3-without-api-key
    try:
        params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % video_id}
        url = "https://www.youtube.com/oembed"
        query_string = urllib.parse.urlencode(params)
        url = url + "?" + query_string

        with urllib.request.urlopen(url) as response:
            response_text = response.read()
            data = json.loads(response_text.decode())
            yt_title = data['title']
            return yt_title
    except Exception as e:
        print("An error occurred while fetching video title:", str(e))
        return None


def train_sentiment_model(comments_df):
    """
    Train sentiment analysis model using the comments DataFrame
    arguments:
        comments_df: comments dataframe with sentiments defined
    returns:
        tuple: tuple with accuracy of the trained model, the best parameters
        after hyperparameter tuning, and confusion matrix of the model
    """
    print("Training the sentiment model model")
    x = comments_df['Comment'].values
    y = comments_df['Sentiment'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.1,
                                                        random_state=10,
                                                        stratify=y)

    print("Length of x_train-", len(x_train))
    print("Length of y_train-", len(y_train))
    print("Length of x_test-", len(x_test))
    print("Length of y_test-", len(y_test))
    text_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', SVC())
    ])

    param_grid = {
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__C': [0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(text_model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    print("Best params:", best_params)
    print("Best estimator:", best_estimator)

    y_pred = best_estimator.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy*100, best_params, conf_matrix


def plot_sentiment_distribution(positive_count, negative_count, neutral_count):
    """
    Plot the distribution of sentiment analysis counts
    arguments:
        positive_count: number of positive sentiment comments in whole comments dataframe
        negative_count: number of negative sentiment comments in whole comments dataframe
        neutral_count: number of neutral sentiment comments in whole comments dataframe
    """
    sentiment = {'Negative': negative_count, 'Neutral': neutral_count, 'Positive': positive_count}

    # Define color palette for sentiment categories
    colors = ['red', 'green', 'blue']

    # Create stacked bar plot using angles and colors
    plt.pie(sentiment.values(), radius=1, labels=sentiment.keys(), autopct="%1.1f%%",
            startangle=140, colors=colors, wedgeprops=dict(width=0.4), pctdistance=0.8)
    plt.title("Sentiment Analysis Distribution")
    plt.axis('equal')

    # Save the plot as an image
    plot_path = "../images/sentiment_pie_chart.png"
    plt.savefig(plot_path)
    plt.clf()


def plot_confusion_matrix(confusion_matrix_data):
    """
    Plot the confusion matrix for the SVM trained model
    arguments:
        confusion_matrix_data: Confusion matrix array containing nine values
    """
    class_names = ['Negative', 'Neutral', 'Positive']
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    # Save the plot as an image file
    plot_path = '../images/confusion_matrix.png'
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(plot_path)
    plt.clf()

def plot_kernel_accuracies(comments_df):
    """
    Plot the accuracies of different kernel types(linear, rbf, poly)
    for different values of C parameter
    arguments:
        comments_df: comments dataframe
    """
    linear_accuracies = []
    rbf_accuracies = []
    poly_accuracies = []
    x_comment = comments_df['Comment'].values
    y_sentiment = comments_df['Sentiment'].values

    # As suggested for Results Feedback, data is unbalanced so I have implemented
    # Random OverSampling to balance the minority class
    # Reference: https://imbalanced-learn.org/stable/over_sampling.html
    ros = RandomOverSampler(random_state=42)
    x, y = ros.fit_resample(x_comment.reshape(-1, 1), y_sentiment)
    x = x.flatten()
    param_grid_accuracy = {
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__C': [0.1, 1, 10, 100]
    }
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.1,
                                                        random_state=10,
                                                        stratify=y)

    # Iterate through each set of hyperparameters
    for kernel in param_grid_accuracy['model__kernel']:
        for c_value in param_grid_accuracy['model__C']:
            text_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('model', SVC(kernel=kernel, C=c_value))
            ])

            # Fit model and make predictions on test data
            text_model.fit(x_train, y_train)
            y_pred = text_model.predict(x_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred) * 100

            # Append accuracies to a list based on kernel type
            if kernel == "linear":
                linear_accuracies.append(accuracy)
            elif kernel == "rbf":
                rbf_accuracies.append(accuracy)
            else:
                poly_accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    # Plot the accuracies for each kernel type
    plt.plot(param_grid_accuracy['model__C'], linear_accuracies, marker='o', label='Linear Kernel')
    plt.plot(param_grid_accuracy['model__C'], rbf_accuracies, marker='o', label='RBF Kernel')
    plt.plot(param_grid_accuracy['model__C'], poly_accuracies, marker='o', label='Poly Kernel')

    plt.title('Accuracies of Different Kernel Types')
    plt.xlabel('C (Regularization parameter)')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.xticks(param_grid_accuracy['model__C'], param_grid_accuracy['model__C'])
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = '../images/kernel_accuracies.png'
    plt.savefig(plot_path)
    plt.clf()


def analyze_video(youtube_api_key, video_url="https://www.youtube.com/watch?v=ZImYRu7hli4"):
    """
    Analyzes sentiment and generates visualizations for a YouTube video
    arguments:
        youtube_api_key: static API key for accessing the YouTube Data API to fetch comments
        video_url: string type URL of the YouTube video with a defaults to a sample video
    returns:
        tuple: tuple containing the video title, accuracy of the sentiment analysis model,
               and the best parameters
    """
    # Extract video ID from the YouTube URL
    video_id = video_url.split("=")[-1]
    print("Video id is : ", video_id)

    # Extract comments using the YouTube API
    comments_df = extract_yt_comments(video_id, youtube_api_key)

    # Clean comments(special char, emoji, space removal)
    cleaned_comments_df = clean_comments(comments_df)

    # Preprocess comments(tokenize and lemmatize)
    preprocessed_comments_df = preprocess_comments(cleaned_comments_df)

    labelled_comments_data, pos, neg, neu = analyze_sentiment(preprocessed_comments_df)

    title = fetch_youtube_title(video_id)

    # Train sentiment model
    accuracy, best_params, conf_matrix = train_sentiment_model(labelled_comments_data)

    # Plot for result analysis
    plot_sentiment_distribution(pos, neg, neu)
    plot_kernel_accuracies(labelled_comments_data)
    plot_confusion_matrix(conf_matrix)

    return title, accuracy, best_params
