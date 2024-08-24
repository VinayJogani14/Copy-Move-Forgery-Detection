from flask import Flask, request, redirect
from datetime import datetime
from dotenv import load_dotenv

from model_files.model import classify
from utils import load_delete_user_images_scheduler

import os

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
upload_path = os.path.join(ROOT_DIR, 'user_images')

app.config['UPLOAD'] = upload_path

load_dotenv()

load_delete_user_images_scheduler()

truthlens_client_url = os.getenv('TRUTHLENS_CLIENT_URL')

@app.route('/')
def home():
    return redirect(truthlens_client_url)

@app.route('/api/classify', methods = ['GET','POST'])
def classify_image():
    if request.method == 'GET':
        return redirect(truthlens_client_url)

    if not request.files.get('image', None):
        return 'No file given for processing'

    if os.path.exists("user_images") == False:
        os.makedirs("user_images")
    
    image_file = request.files['image']

    image_file_name = datetime.now().strftime("%d-%m-%Y%H%M%S") + image_file.filename
    image_file.save(os.path.join(app.config['UPLOAD'], image_file_name))

    input_image_path = os.path.join(ROOT_DIR, "user_images", image_file_name)
    prediction = classify(input_image_path)
    if prediction == 0:
        return { "prediction": 0 }
    elif prediction == 1:
        return { "prediction": 1 }
    else:
        return { "error": 'An Error has occurred' }

if __name__ == "__main__":
    app.run()
