from flask import Flask ,render_template,request
import numpy as np
import pickle

with open ('model.pkl','rb') as file1:
    mdl =pickle.load(file1)

with open ('cv.pkl','rb') as file2:
    cv_mdl =pickle.load(file2)
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data',methods=['GET','POST'])
def data():
    class_list=['business', 'entertainment', 'politics', 'sport', 'tech']

    article = request.form['text']
    text = ["".join(article)]
    user_count_vec =cv_mdl.transform(text)
    resul=mdl.predict(user_count_vec)
    results=class_list[resul[0]]
    return render_template('index.html',result=results)


if __name__ == "__main__":
    app.run(host = '127.0.0.100',debug=True)