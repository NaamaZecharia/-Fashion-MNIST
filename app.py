import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.preprocessing.image import load_img
from werkzeug.utils import secure_filename

#the predicted model
from model import process_image, predict_class 

app = Flask(__name__)

photos = UploadSet('photos',IMAGES)

#path for saving the images
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos) 



@app.route('/home', methods=['GET', 'POST'])

def hello_world():
    return "Hello, World!"

@app.route('/upload', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return 'There is no photo in form!'
        #filename = request.args.get('photo')
        # save the image
        filename = secure_filename(request.files['photo'].filename)
        filename = photos.save(request.files['photo'])
        # load the image
        image = load_img('./static/img/'+filename, grayscale=True, target_size=(28,28))
        # process the image
        image = process_image(image)
        # make prediction
        prediction, percentage = predict_class(image)
        # the answer which will be rendered back to the user
        answer = "For {} : <br>classified as class: {} with propability of: {}% ".format(filename, prediction, percentage)
        return answer
    # web page to show before the POST request containing the image
    return render_template('upload.html')

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8080)


