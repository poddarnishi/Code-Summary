from flask import Flask, request, render_template, send_file
from functions import summarize_code, summarize_file
import os
import glob
from zipfile import ZipFile
app = Flask(__name__)

@app.route('/',methods=["POST","GET"])
def home():
    decision=False
    if request.method=="POST":
        print("file uploaded")
        f=request.files["myFile"]
        f.save(f.filename)
        with ZipFile(f,'r') as f1:
            f1.extractall(path="files-from-zip")
        decision=summarize_file()
    return render_template('index.html', decision=decision)
@app.route('/download')
def download_file():
    file_path = 'summary.txt'

    return send_file(file_path, as_attachment=True)
app.run(debug=True)
@app.route('/code_summary',methods=["POST","GET"])
def code():
    if request.method=="POST":
        if "python_code" in request.form:
            python_code = request.form['python_code']
            with open('files-for-code/user_code.py', 'w') as f:
                f.write(python_code)
            message=summarize_code()
            return render_template('summary.html', messages=message,python=python_code)
    else:
        return render_template('summary.html')