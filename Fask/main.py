from flask import *
import pandas as pd

import sklearn

import scipy

import numpy as np

import matplotlib as mp

from scipy import stats

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split as tts

from sklearn.neighbors import KNeighborsClassifier
app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def home():
	if request.method == "POST":
		f = request.files['attachment']
		f.save("static/sample.csv")
		d=pd.read_csv(r"static/sample.csv")
		d.dropna(inplace=True)
		x=d.iloc[:,:-1]
		y=d.iloc[:,-1]
		xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.20)
		model=KNeighborsClassifier(n_neighbors=1)
		model.fit(xtrain,ytrain) # to train the machine
		ypred=model.predict(xtest)
		#print(model)
		print(accuracy_score(ytest, ypred))
		Output = (model.predict([[5.1,3.5,1.4,0.2]]))
		Negative = ['ajsdfhkasd']
		Neutral = ['kjdhfkh']
		return render_template("index.html",positive=Output, negative=Negative, neutral=Neutral, response=1)
	return render_template("index.html",data="", response=0)

if __name__ == '__main__':
	app.run(debug=True)