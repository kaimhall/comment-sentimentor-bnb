from flask import Flask, request, render_template
import pandas as pd
import joblib
import sklearn.feature_extraction.text as text

# declare a Flask app
app = Flask(__name__)

# router
@app.route('/', methods=['GET', 'POST'])

def Main():

  if request.method == 'POST':
    tf = clf = joblib.load("vectorizer.pkl")
    clf = joblib.load("NBbayes.pkl")

    # get sentence
    sentence = request.form.get("sentence")

    # predict
    prediction = clf.predict(
      tf.transform([sentence])
    )
  
  else:
    prediction = ""
  output = 'insulting' if prediction == 1 else 'not insulting'
  return render_template("NaiveBayes.html", output = f'comment is {output}')

if __name__ == '__main__':
    app.run(debug = True)