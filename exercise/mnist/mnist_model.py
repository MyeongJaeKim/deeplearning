import os
# 텐서플로우 디버그 로그 끔
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter

tf.reset_default_graph()


class MNISTModel:
    def __init__(self, model_path='./mnist_model', prediction_tensor_name='predictor'):
        # Path Parameters
        self.model_path = model_path

        # 모델 복구 및 세션 초기화
        self._load_model()
        self._setup_tf_session()

        self.prediction_tensor_name = prediction_tensor_name

        self.input_tensor_name = "X:0"
        self.dropout_rate_name = "keep_prob:0"

    # 저장된 Model Load
    def _load_model(self):
        self.saved_model = tf.train.get_checkpoint_state(self.model_path)

        if not self.saved_model or not self.saved_model.model_checkpoint_path:
            print("Could not find MNIST prediction model")
            exit(-1)

        meta_graph_path = self.saved_model.model_checkpoint_path + ".meta"
        self.saved_graph = tf.train.import_meta_graph(meta_graph_path)

    # 텐서플로우 세션 생성
    def _setup_tf_session(self):
        self.sess = tf.Session()
        self.saved_graph.restore(self.sess, self.saved_model.model_checkpoint_path)
        self.predictor = tf.get_collection('predictor')[0]
        print("MNIST Model was successfully loaded")

    # Raw bytes 를 28 by 28 normalized array 로 변환
    def _convert_to_mnist_pixel(self, img_array):
        im = Image.fromarray(img_array).convert('L')
        #im = Image.frombytes(mode='L', data=img_array, size=(140, 140))
        width = float(im.size[0])
        height = float(im.size[1])
        new_image = Image.new('L', (28, 28), 255)

        # Width 가 더 크면
        if width > height:
            # 가로 세로 비율에 따라 Resize --> 20 x 20
            height = int(round((20.0 / width * height), 0))

            if height == 0:
                height = 1

            img = im.resize((20, height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - height) / 2), 0))

            new_image.paste(img, (4, wtop))
        else:
            width = int(round((20.0 / height * width), 0))

            if width == 0:
                width = 1

            img = im.resize((width, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - width) / 2), 0))
            new_image.paste(img, (wleft, 4))

        tv = list(new_image.getdata())

        # Pixel 값을 Normalize (0 ~ 1)
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        return tva

    def do_predict(self, image_raw_bytes, image_processing=True):
        # let's do something
        if image_processing:
            image = self._convert_to_mnist_pixel(image_raw_bytes)
        else:
            image = image_raw_bytes

        feed_dict = {self.input_tensor_name: image, self.dropout_rate_name: 1.0}
        y_hat = self.sess.run(self.predictor, feed_dict=feed_dict)

        predicted_number = np.argmax(y_hat)
        probability = '{:04.2f}'.format(np.max(y_hat) * 100)

        print("Prediction was made, result : ", predicted_number, ", Probability was ", probability)

        return predicted_number, probability




