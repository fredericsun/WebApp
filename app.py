from flask import Flask, render_template, request
from utils import process_image, deprocess_image
from models import get_evaluate_model
import numpy as np
import cv2 as cv
from tensorflow import Graph, Session
import uuid
import shutil
import os


class KerasModel(object):

    def __init__(self, path):
        self.graph = Graph()
        self.session = Session(graph=self.graph)
        self.eval_model = None
        self.load_model(path)

    def load_model(self, path):
        with self.graph.as_default(), self.session.as_default():
            self.eval_model = get_evaluate_model()
            self.eval_model.load_weights(path)

    def predict(self, content):
        with self.graph.as_default(), self.session.as_default():
            res = self.eval_model.predict([content])
            return res


# OpenCV GrabCut
def grabcut(image, coordinate):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = tuple(coordinate)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    mask_output = mask2[:, :, np.newaxis]
    return image, mask_output


def predict(model, img_read):
    # Read image
    content = process_image(img_read)

    ori_height = content.shape[1]
    ori_width = content.shape[2]

    # Pad image
    content = get_padding(content)
    height = content.shape[1]
    width = content.shape[2]

    # Generate output and save image
    res = model.predict(content)
    output = deprocess_image(res[0], width, height)
    output = remove_padding(output, ori_height, ori_width)
    return output


def get_padding(image, axis_expanded=True):
    height = image.shape[1]
    width = image.shape[2]
    pad_height = (height // 8 + 1) * 8 - height
    pad_width = (width // 8 + 1) * 8 - width
    if axis_expanded:
        padding = (0, 0), (0, pad_height), (0, pad_width), (0, 0)
    else:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    new_image = np.pad(image, padding, 'reflect')
    return new_image


def remove_padding(image, ori_height, ori_width):
    new_image = image[0:ori_height, 0:ori_width, :]
    return new_image


class FlaskApp(object):
    def __init__(self):
        self.eval_model_1 = KerasModel('./trained_nets/1.h5')
        self.eval_model_2 = KerasModel('./trained_nets/2.h5')
        self.eval_model_3 = KerasModel('./trained_nets/3.h5')
        self.eval_model_12 = KerasModel('./trained_nets/12.h5')
        self.eval_model_13 = KerasModel('./trained_nets/13.h5')
        self.eval_model_23 = KerasModel('./trained_nets/23.h5')
        self.trained_nets = {'1': self.eval_model_1, '2': self.eval_model_2, '3': self.eval_model_3,
                             '12': self.eval_model_12, '13': self.eval_model_13, '23': self.eval_model_23}

        self.app = Flask(__name__, static_url_path='/static')

        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            if request.method == 'POST':
                shutil.rmtree('./static/img/output/')
                os.makedirs('./static/img/output/')
                image = request.files['file_photo']
                image = np.fromstring(image.read(), np.uint8)
                image = cv.imdecode(image, cv.IMREAD_UNCHANGED)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                style = request.form.getlist('style')
                style = ''.join(style)

                coordinate = request.form.getlist('output_point')

                # partial transfer
                if coordinate[0] is not '':
                    image_cut, mask = grabcut(image, list(map(int, coordinate[0].split(","))))
                    image_trans = predict(self.trained_nets[style], image)
                    res = mask * image_trans + -(mask - 1) * image
                else:
                    image_trans = predict(self.trained_nets[style], image)
                    res = image_trans

                rand_name = str(uuid.uuid4()).replace('-', '')[0:15]
                cv.imwrite('./static/img/output/output_' + rand_name + '.jpg', res)
                return render_template('results.html', result_name='./static/img/output/output_' + rand_name + '.jpg')
            else:
                return render_template('index.html')

    def get_app(self):
        return self.app


flask_app = FlaskApp()
app = flask_app.get_app()

if __name__ == '__main__':
    app.run(debug=False)
