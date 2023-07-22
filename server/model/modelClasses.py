import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional, Flatten, BatchNormalization
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import re

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
    
