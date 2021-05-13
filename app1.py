from flask import Flask, render_template, request
import pickle
import numpy as np
a=Flask(__name__)

model = pickle.load(open('model_wcat.pkl', 'rb'))


@a.route("/")
def home():
    return render_template("home.html")

@a.route("/predict",methods=['POST'])
def predict():

        var3=[float(x) for x in request.form.values()]
        var4=[np.array(var3)]
        #var2=open("model1.pkl","rb")
        #var3=joblib.load(var2)
        var5=model.predict(var4)
        var6=round(float(var5),2)

        return render_template("predict.html",pred=var6)


if __name__=="__main__":
   a.run(debug=True)
