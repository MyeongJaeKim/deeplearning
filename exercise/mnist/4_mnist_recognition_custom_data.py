import os
import tensorflow as tf
import numpy as np
from exercise.mnist import image_reader

# 텐서플로우 디버그 로그 끔
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.reset_default_graph()

# 파라미터
MODEL_DIR = './mnist_model'
CUSTOM_IMG_DIR = './custom_data'
CUSTOM_IMG_EXT = '.png'

# Evaluation 에 사용할 커스텀 Data Set 준비
# CUSTOM_IMG_DIR 에서 CUSTOM_IMG_EXT 확장자인 파일 모두 읽는다.
img_paths = []
for file in os.listdir(CUSTOM_IMG_DIR):
    if file.endswith(CUSTOM_IMG_EXT):
        img_paths.append(os.path.join(CUSTOM_IMG_DIR, file))

images = []
for path in img_paths:
    print("Reading Image at " + path)
    images.append(image_reader.read_image(path))

# 텐서플로우 모델 로딩
checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)

if not checkpoint or not checkpoint.model_checkpoint_path:
    print("Could not find MNIST prediction model")
    exit(-1)

model_path = checkpoint.model_checkpoint_path + ".meta"
saved_graph = tf.train.import_meta_graph(model_path)

with tf.Session() as sess:
    # sess 세션에 저장된 모델을 읽는다.
    saved_graph.restore(sess, checkpoint.model_checkpoint_path)
    print("Model was successfully loaded: ", checkpoint.model_checkpoint_path)

    # 복구된 모델의 Tensor Name 검색
    #g = tf.get_default_graph()
    #nodes = g.get_all_collection_keys()
    #print([i.name for j in nodes for i in g.get_collection(j)])

    feed_dict = {"X:0": images, "keep_prob:0": 1.0}

    prediction = tf.get_collection('predictor')[0]
    y_hat = sess.run(prediction, feed_dict=feed_dict)

    for i in range(len(y_hat)):
        print((i + 1), "th prediction is ", np.argmax(y_hat[i]))
