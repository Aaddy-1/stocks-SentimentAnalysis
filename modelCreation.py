# All of our imports
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional, Flatten, BatchNormalization
import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# This function cleans comments by removing stopwords, lemmatizing words, removing links and emojis

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

print(commentCleaner(["One of the other reviewers mentioned watching 1 oz episode"]))

# Tokenizes and pads comments for feeding into our keras model

def tokenizeComments(comments, tokenizer):
    print("Comments recieved for tokenization: ")
    print(comments)
    print("Fitted tokenizer to sample texts")
    tokenized_comments = tokenizer.texts_to_sequences(comments)
    print("Converted to sequences")
    tokenized_comments = pad_sequences(tokenized_comments, 235)
    print("Padded succesfully")
    print(tokenized_comments)
    return tokenized_comments

# The model we are going to use

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

data1 = pd.read_csv('modelTrainingData/stock_data.csv')

def changeNegativetoZero(val):
    if val == -1:
        return 0
    return val

# changeNegativetoZero = {'-1' : '0', '1' : '1'}
data1['Sentiment'] = data1['Sentiment'].apply(changeNegativetoZero)

data2 = pd.read_csv('modelTrainingData/sent_train.csv')
print(data2['label'].unique())
# We are going to drop all neutral rows
data2.rename(columns={'text': 'Text', 'label': 'Sentiment'}, inplace=True, errors='raise')
data2 = data2.drop(data2[data2['Sentiment'] == 2].index)
print(data2['Sentiment'].unique())

data3 = pd.read_csv('modelTrainingData/sent_valid.csv')
print(data3['label'].unique())
# We are going to drop all neutral rows
data3.rename(columns={'text': 'Text', 'label': 'Sentiment'}, inplace=True, errors='raise')
data3 = data3.drop(data3[data3['Sentiment'] == 2].index)
print(data3['Sentiment'].unique())

data4 = pd.read_csv('modelTrainingData/sentiment.csv')

# We are going to remove tweet url and stock ticker columns
mapping = {'Negative' : 0, 'Positive': 1}
data4['Sentiment'] = data4['Sentiment'].map(mapping)

data4 = data4.drop(columns=['Stock Ticker', 'Tweet URL'])
data4.rename(columns={'Tweet Text': 'Text'}, inplace=True, errors='raise')

data5 = pd.read_csv('modelTrainingData/augmented_data.csv')
data5 = data5.drop(columns=['Unnamed: 0'])
data5 = data5.drop(data5[data5['Sentiment'] == 'neutral'].index)
# data5['Sentiment'].unique()
data5.rename(columns={'Sentence': 'Text', 'Sentiment': 'Sentiment'}, inplace=True, errors='raise')


print(data1.head())
print(data2.head())
print(data3.head())
print(data4.head())
print(data5.head())

from textattack.augmentation import EasyDataAugmenter

import random

augmenter = EasyDataAugmenter()

def augment_text(sentence):
    augmented_sentences = augmenter.augment(sentence)
    if augmented_sentences:
        return random.choice(augmented_sentences)
    else:
        return sentence 

def augmentDataFrame(df):
    augmented_data = []
    augmented_labels = []
    count = 0
    for sentence, sentiment in zip(df.Text, df.Sentiment):
        random_num = random.randint(1, 100)
        if (random_num >= 30 and sentiment == 0):
            augmented_sentence = augment_text(sentence)
            augmented_data.append(augmented_sentence)
            augmented_labels.append(sentiment)
            count += 1
        if (count > 2554):
            break
    
    new_df = {"Text" : augmented_data, "Sentiment" : augmented_labels}
    new_df = pd.DataFrame(new_df)

    df = pd.concat([df, new_df])
    # df = df.append(pd.DataFrame(new_df))
    return df

final_data = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)
print(final_data.head())
print(final_data.info())
print(final_data['Sentiment'].unique())

def changeStringToNum(val):
    if val == 'negative':
        return 0
    elif val == 'positive':
        return 1
    return val

final_data['Sentiment'] = final_data['Sentiment'].apply(changeStringToNum)
print(final_data['Sentiment'].unique())
print(final_data.head())

final_data.to_csv('final_data.csv')

checkpoint_path = "trial1/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
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

class textTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        print("Starting fitting")
        return self
    
    def transform(self, X, y=None):
        print("Starting transform")
        print(X)
        # tokenizerFinal = Tokenizer(num_words=1000, split=' ') 
        # print(cleaned_data['Sentence'].values)
        # tokenizerFinal.fit_on_texts(cleaned_data['Sentence'].values)
        X_cleaned = commentCleaner(X)
        print("Cleaned comments")
        print("Starting tokenization")
        X_tokenized = tokenizeComments(X_cleaned, self.tokenizer)
        print("Tokenized")
        print("Ending transform")

        return X_tokenized
    
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

print(final_data.shape)
print(final_data.head())
final_data = augmentDataFrame(final_data)
print(final_data.shape)

model = customModel(8)
X = final_data['Text']
y = pd.get_dummies(final_data['Sentiment'])

tokenizer = Tokenizer(num_words=18364, split = ' ')
x = final_data['Text'].values
print(x)
x = commentCleaner(x)
print(x)
tokenizer.fit_on_texts(x)

pipeline = Pipeline(steps=[('textTransform', textTransformer(tokenizer = tokenizer)), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=20, stratify=y)

pos_count = y_train.sum()
print('Length: ', len(y_train))
print('count: ', pos_count)

print(model.model_fn.summary())

pipeline.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have the loaded model and testing data
# model = your_loaded_model
# X_test = testing_text_inputs
# y_test = true_sentiment_labels
# print(y_train)
# pipeline.named_steps['model'].model.load_weights('final1/weights-improvement-05-0.1616.hdf5')
y_pred = pipeline.predict(X_test)

# print(conf_matrix)

# Convert the predictions to binary values based on a threshold (e.g., 0.5)
print(y_pred)

y_pred_binary = (y_pred[:, 1] > 0.5).astype(int)
y_test_binary = np.argmax(y_test, axis=1)
print(y_pred_binary)
# Calculate evaluation metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

import pickle
import os

def save_pipeline_keras(model,folder_name="model"):
    os.makedirs(folder_name, exist_ok=True)
    # print(model.named_steps['transformText'])
    # model.named_steps['model'].model = None
    # dump(model, 'pipeline.pkl')
    pickle.dump(model.named_steps['textTransform'], open(folder_name+'/'+'textTransformer.pkl','wb'))
    pickle.dump(tokenizer, open(folder_name + '/' + 'tokenizer.pkl', 'wb'))
    model.named_steps['model'].model.save(folder_name+'/model.h5')
    # pickle.dump(model.named_steps['model'].model, open(folder_name + '/' + 'model.h5', 'wb'))

save_pipeline_keras(pipeline, 'finalPipeline')