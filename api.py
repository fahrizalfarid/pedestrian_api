import os
from flask import Flask,request,Response,jsonify
from flask_limiter import Limiter
from flask_cors import CORS
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from helper.helper import helper



helper = helper()
model = helper.loadModel()


UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','webp'])


app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["10 per second"]
)




@app.route('/api/image/<filter>', methods=["POST"])
def predictImage(filter):
    r = request

    if r.method == 'POST':
        if 'image' not in r.files:
            return jsonify({
                'status':{
                    'code':301,
                    'description':'No image part.'
                }
            })

        file = r.files['image']

        if file.filename == '':
            return jsonify({
                'status':{
                    'code':302,
                    'description':'No selected image.'
                }
            })

        

        if file and file.filename.split('.')[1] in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            image, counter, confidence = helper.predictToModel(
                model, 'images/'+filename, filter
            )

            if confidence != 0:
                imageResized = helper.imageConverter(
                    0, helper.imageReducers(image)
                )

                return jsonify({
                    'status':{
                        'code':200,
                        'description':'Success.'
                    },
                    'predicted':{
                        'label':filter,
                        'image':imageResized,
                        'confidence':confidence,
                        'count':counter

                    }
                })
        else:
            return jsonify({
                'status':{
                    'code':303,
                    'description':'Extension does not allow.'
                }
            })
                
    return jsonify({
        'status':{
            'code':304,
            'description':'No Detection.'
        },
        'predicted':{
            'label':filter,
            'image':'null',
            'confidence':'null',
            'count':'null'

        }
    })
        




if __name__ == "__main__":
    app.run(host="127.0.0.1",port=6000, debug=True)