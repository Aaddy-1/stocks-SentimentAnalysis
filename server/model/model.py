from sklearn.pipeline import Pipeline
import keras
import dill as pickle
from modelClasses import textTransformer, customModel
import numpy as np
from pathlib import Path

def load_pipeline_keras(cleaner, model, tokenizer, folder_name="model"):
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
    classifier = load_pipeline_keras('/Users/aadeesh/redditSentiment/server/model/classifier/textTransformer.pkl', 
                    '/Users/aadeesh/redditSentiment/server/model/classifier/model.h5', 
                    '/Users/aadeesh/redditSentiment/server/model/classifier/tokenizer.pkl', 
                    'server/model/classifier')
    return classifier



def classifier_predict(classifier, text):
    return np.argmax(classifier.predict(text), axis = 1)

# classifier_predict(['I love this money', 'I lost a lot of money'])
