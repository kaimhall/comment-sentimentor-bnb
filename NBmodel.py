# imports
#------------------------------------------
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection as ms
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt
import joblib

# data
#--------------------------------------------
df = pd.read_csv(
  'https://github.com/ipython-books/''cookbook-2nd-data/blob/master/''troll.csv?raw=true'
)

# model Bernoulli naive bayes
#--------------------------------------------
y = df['Insult']

tf = text.TfidfVectorizer()  # text frequency vector = binary vector for each unique word
X = tf.fit_transform(df['Comment'])

joblib.dump(tf, "vectorizer.pkl")

(X_train, X_test, y_train, y_test) = ms.train_test_split(X, y, test_size=.2)

bnb = ms.GridSearchCV(
    nb.BernoulliNB(),
    param_grid={'alpha': np.logspace(-2., 2., 50)}
    )

bnb.fit(X_train, y_train)
score = bnb.score(X_test, y_test)

joblib.dump(bnb, "NBbayes.pkl")