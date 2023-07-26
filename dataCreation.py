import pandas as pd
from pathlib import Path
SERVER_FOLDER = Path(__file__).parent.parent.resolve()
comments = pd.read_csv("server/Data/redditData/comment.csv")
len(comments.index)

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional, Flatten, BatchNormalization
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re
import numpy as np

def classification_model():
    # Building our model
    model = keras.Sequential()
    model.add(Embedding(18364, 256, input_length = 235))
    model.add(SpatialDropout1D(0.5))
    
    model.add(Bidirectional(LSTM(units=128, dropout=0.6)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2,activation='softmax'))


    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

checkpoint_path = "final1/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# Create a ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='loss',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)

# Create an EarlyStopping callback to stop training if validation loss doesn't improve
early_stopping_callback = EarlyStopping(
    monitor='loss',
    patience=5,  # Number of epochs with no improvement after which training will stop
    verbose=1
)


class customModel(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size):
        self.model_fn = classification_model()
        self.batch_size = batch_size
        self.model = self.model_fn
    
    def fit(self, X, y):
        
        with tf.device('/device:GPU:0'):
            self.model.fit(X, y, epochs = 7, batch_size=self.batch_size, callbacks = [checkpoint_callback, early_stopping_callback], verbose = 1)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

def commentCleaner(comments):
    cleaned_comments = []
    for comment in comments:
        # Remove special symbols, emojis, reddit username mentions, and hyperlinks
        comment = re.sub(r"[^\w\s]|http\S+|www\S+|u/[A-Za-z0-9_-]+", "", comment)
        comment = comment.lower()
        # Tokenize the comment
        tokens = comment.split()
        # tokens = comment.split(' ')
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join the tokens back into a single string
        cleaned_comment = " ".join(tokens)
        cleaned_comments.append(cleaned_comment)   
    return cleaned_comments


    
def tokenizeComments(comments, tokenizer):
    # print("Comments recieved for tokenization: ")
    # print(comments)
    # print("Fitted tokenizer to sample texts")
    tokenized_comments = tokenizer.texts_to_sequences(comments)
    # print("Converted to sequences")
    tokenized_comments = pad_sequences(tokenized_comments, 235)
    # print("Padded succesfully")
    # print(tokenized_comments)
    return tokenized_comments

class textTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        # print("Starting fitting")
        return self
    
    def transform(self, X, y=None):
        # print("Starting transform")
        # print(X)
        # tokenizerFinal = Tokenizer(num_words=1000, split=' ') 
        # print(cleaned_data['Sentence'].values)
        # tokenizerFinal.fit_on_texts(cleaned_data['Sentence'].values)
        X_cleaned = commentCleaner(X)
        # print("Cleaned comments")
        # print("Starting tokenization")
        X_tokenized = tokenizeComments(X_cleaned, self.tokenizer)
        # print("Tokenized")
        # print("Ending transform")

        return X_tokenized
    
import dill as pickle

def load_pipeline_keras(cleaner, model, tokenizer):
    cleaner = pickle.load(open(cleaner,'rb'))
    tokenizerFinal = pickle.load(open(tokenizer,'rb'))
    model = keras.models.load_model(model)
    cleaner.tokenizer = tokenizerFinal
    # classifier = KerasClassifier(model=build_model, epochs=1, batch_size=10, verbose=1)
    # classifier.classes_ = pickle.load(open(folder_name+'/'+classes,'rb'))
    # classifier.model = build_model
    # build_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return Pipeline([
        ('textTransformer', cleaner),
        ('model', model)
    ])


def init_model():
    classifier = load_pipeline_keras('finalPipeline/textTransformer.pkl', 
                    'finalPipeline/model.h5', 
                    'finalPipeline/tokenizer.pkl', 
                    )
    return classifier

classifier = init_model()


def dataframeProcessor(df, classifier):

    keywords = {"Tesla" : ["$tsla", "tsla", "tesla", "elon musk", "musk"],
            "Apple" : ["$aapl", "aapl", "apple", "mac", "iphone", "airpods", "macbook"], 
            "Nvidia" : ["$nvda", "nvda", "nvidia", "rtx", "geforce", "jensen", "huang"], 
            "Google" : ["$googl", "googl", "google", "alphabet", "bard", "android", "pixel", "sundar pichai", "sundar", "pichai"],
            "Amazon" : ["$amzn", "amzn", "amazon", "aws", "prime", "alexa", "fire tv", "amazon prime"],
            "Microsoft" : ["$msft", "msft", "microsoft", "windows", "azure", "xbox"],
            "Meta" : ["$meta", "meta", "instagram", "facebook", "threads"]
        }
    keywords2 = ["$tsla", "tsla", "tesla", "elon musk", "musk", 
             "$aapl", "aapl", "apple", "mac", "iphone", "airpods", "macbook"
             "$nvda", "nvda", "nvidia", "rtx", "geforce", "jensen huang", "jensen", "huang" 
             "$googl", "googl", "google", "alphabet", "bard", "android", "pixel", "sundar pichai", "sundar", "pichai"
             "$amzn", "amzn", "amazon", "aws", "prime", "alexa", "fire tv", "amazon prime"
             "$msft", "msft", "microsoft", "windows", "azure", "xbox"
             "$meta", "meta", "instagram", "facebook", "threads"
        ]

    filtered_df = df[df['Comment'].str.contains('|'.join(keywords2), case = False)]

    # Add an extra column to the filtered dataframe that indicates which keyword was present in that comment
    def keyWordBuilder(comment):
        returnString = ""
        for keyword in keywords2:
            if keyword in comment.lower():
                for key in keywords:
                    if keyword in keywords[key]:
                        if key not in returnString:
                            returnString += key + ' '
        if returnString == "":
            return "None"
        return returnString

    keyWordList = filtered_df['Comment'].apply(keyWordBuilder)

    filtered_df = filtered_df.assign(Keyword = keyWordList)

    newDates = pd.to_datetime(filtered_df['Date'])
    newDates = newDates.dt.date
    filtered_df = filtered_df.assign(Date = newDates)
    filtered_df = filtered_df.sort_values(by='Date', ascending=True)

    comments = filtered_df.Comment
    preds = classifier.predict(comments)

    sentiments = np.argmax(preds, axis = 1)
    # preds

    filtered_df = filtered_df.assign(Sentiment = sentiments)

    return filtered_df
    
    
processed_df = dataframeProcessor(comments, classifier=classifier)
print(processed_df.head())
processed_df.to_csv('server/Data/redditData/Posts/processed_df.csv')