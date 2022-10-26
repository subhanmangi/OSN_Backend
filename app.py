from flask import *
import pandas as pd
import json
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import tweepy as tw
#nltk.download('wordnet')
#nltk.download('all')

"""
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')
"""

apikey = 'Y4PcDl6rVWDEZ1XlzekuZTsGQ'
apisecretkey = 'vDKxuLeRBQJfoQHJsQsiN5wO2sFAGhrZGSWb61DZzwB9NaURth'
bearerToken = 'AAAAAAAAAAAAAAAAAAAAADBqiAEAAAAA0nA93ASsenuRyLtIdGW6wDWcY%2Fo%3DzOY7RfhuoDWcwWkNhVB2pwIBPYa0x3fknqFiGd96bLQMFgBeTQ'
accesstoken = '2882123819-zdnp2sn9ZaRjitq8WL3HtBdyvfkqXfGHjDcONqS'
accesstokensecret = 'TFVF1IKQ1fG8tcpfCPcpcrliyEISfkFJEIRWGvxjOWnEQ'

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)

app = Flask(__name__)




# change i'll to i will, won't to will not and so on
def decontracted(phrase):

    phrase = re.sub(r"\’", "\'", phrase)
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"shoudn\'t", "should not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"don\'t", "do not", phrase)
    phrase = re.sub(r"doesn\'t", "doesn not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# clean the tweets by removing mentions, hashtag symbols and links
def analyze_tweets(contents):
    contents = decontracted(contents)

    # remove any stop words present
    contents = ' '.join([item for item in contents.split() if item not in stop])
    # remove any stop words present
    contents = ''.join(ch for ch in contents if ch not in exclude)
    contents = re.sub(r'@[A-Za-z0-9_]+', '', contents)

    # Removing Hyperlinks (if any)
    contents = re.sub(r"(https?://[^\s]+)", '', contents)
    ## Removing digits
    contents = ''.join(i for i in contents if not i.isdigit())

    contents = re.sub(r"[-()\'\"/;:<>{}`+=~|.!?,…—]", "", contents)
    # Expanding contractions

    contents = decontracted(contents)
    # contents = contents.lower().strip()
    return contents

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

@app.route('/search/', methods=['GET'])
def api_sentinet():
    user_query = str(request.args.get('query')).split(',')

    auth = tw.OAuthHandler(apikey, apisecretkey)
    auth.set_access_token(accesstoken, accesstokensecret)
    api = tw.API(auth, wait_on_rate_limit=False)
    query = ""
    for i in range(len(user_query)):
        if(i!=len(user_query)-1):
            query = query + user_query[i] + " OR "
        else:
            query = query + user_query[i]
    #query = "covid OR corona OR covid19"
    #print(query)
    limit = 3
    tweets = tw.Cursor(api.search_tweets, q=query, tweet_mode='extended').items(limit)
    import pandas as pd
    columns = ['TweetId', 'User', 'Tweet']
    data = []
    for tweet in tweets:
        data.append([tweet.id, tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)
    df['clean_tweet'] = df['Tweet'].apply(analyze_tweets)

    sentiment_list = []
    for sentence in df['clean_tweet']:
        # sentence='covid19 is very dangerous!'
        token = nltk.word_tokenize(sentence)
        after_tagging = nltk.pos_tag(token)
        #print(token)
        #print(after_tagging)
        tokens_count = 0
        sentiment = 0.0
        # tokens_count = 0
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        for word, tag in after_tagging:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            #print(swn_synset)

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
        if sentiment > 0:
            sentiment_list.append('positive')
        elif sentiment < 0:
            sentiment_list.append('negative')
        else:
            sentiment_list.append('neutral')

    df['sentiment'] = sentiment_list

    dataset = []

    for i in range(len(df.values)):
        row = {
            df.columns[1]: df.values[i][1],
            df.columns[2]: df.values[i][2],
            df.columns[4]: df.values[i][4]}
        dataset.append(row)
    dataset

    return json.dumps(dataset)

@app.route('/search/tb/', methods=['GET'])
def api_tb():
    user_query = str(request.args.get('query')).split(',')

    auth = tw.OAuthHandler(apikey, apisecretkey)
    auth.set_access_token(accesstoken, accesstokensecret)
    api = tw.API(auth, wait_on_rate_limit=False)
    query = ""
    for i in range(len(user_query)):
        if(i!=len(user_query)-1):
            query = query + user_query[i] + " OR "
        else:
            query = query + user_query[i]
    #query = "covid OR corona OR covid19"
    #print(query)
    limit = 3
    tweets = tw.Cursor(api.search_tweets, q=query, tweet_mode='extended').items(limit)
    import pandas as pd
    columns = ['TweetId', 'User', 'Tweet']
    data = []
    for tweet in tweets:
        data.append([tweet.id, tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame(data, columns=columns)
    df['clean_tweet'] = df['Tweet'].apply(analyze_tweets)

    from textblob import TextBlob
    blob = TextBlob(df['clean_tweet'][0])
    sentiments = []

    for tweet in df['clean_tweet']:
        p = float(TextBlob(tweet).sentiment.polarity)
        if p > 0.0:
            sentiments.append('positive')
        elif p < 0.0:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')

    df['sentiment'] = sentiments

    dataset = []

    for i in range(len(df.values)):
        row = {
            df.columns[1]: df.values[i][1],
            df.columns[2]: df.values[i][2],
            df.columns[4]: df.values[i][4]}
        dataset.append(row)
    dataset

    return json.dumps(dataset)

if (__name__=='__main__'):
    app.run(port=7777)
