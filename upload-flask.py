from flask import Flask, render_template, request
import pandas as pd
import os
import model

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv','xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Parameters:
        filename (str): The name of the uploaded file.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_excel(file_path):
    """
    Process the uploaded Excel file.

    Parameters:
        file_path (str): The path of the uploaded Excel file.

    Returns:
        pd.DataFrame: The content of the Excel file as a DataFrame.
    """
    try:
        df = pd.read_excel(file_path)
        model.train_model()
        df1= model.predict_model(df)
        return df1
    except Exception as e:
        #TODO: Handle any exception that might occur during file processing
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle the file upload and display its content.

    Returns:
        render_template: Renders the 'index.html' template with the uploaded data.
    """
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = process_excel(file_path)
            return render_template('index.html', data=data.to_html())
        else:
            return "Invalid file format. Please upload a valid Excel file."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='80',debug=True)
