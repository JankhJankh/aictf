import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
import numpy as np
from sklearn.cluster import KMeans
import sys
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from keras.applications.mobilenet import decode_predictions, preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import io
from keras.backend import clear_session
from imagehash import phash
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import urllib.parse
from PIL import ImageFilter
import nltk
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import base64
import pickle
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from base64 import b64encode
import re

ALLOWED_EXTENSIONS = set(['png'])

warnings.filterwarnings('ignore')
# Probably not needed for production, but I have GPU support enabled on my version
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

THRESHOLD = 0.80
PHASH_TRESH = 2
IMAGE_DIMS = (224, 224)

app = Flask(__name__)

# Heavily taken from https://blog.keras.io/category/tutorials.html
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(IMAGE_DIMS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image


def get_predictions(image):
    clear_session()
    model = load_model("./static/cnn/model.h5")
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


@app.route("/predict", methods=["POST"])
def predict():
    base_img = Image.open("./static/cnn/img/trixi.png").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("frog"):

            frog_img = Image.open(io.BytesIO(request.files["frog"].read()))
            s1 = str(phash(frog_img))
            s2 = str(phash(base_img))

            frog_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            frog_mat = prepare_image(frog_img)

            # read the image in PIL format
            frog_label, frog_conf, top_preds = get_predictions(frog_mat)

            res = {}
            res["is_frog"] = "tree_frog" in frog_label
            res["frog_conf"] = frog_conf
            res["frog_cat"] = frog_label
            res["frog_img_sim"] = frog_dist
            res["top_5"] = top_preds
            res["backpage"] = "cnn.html"
            if "tree_frog" in frog_label and frog_conf >= THRESHOLD and frog_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{Default CNN}", results=res)
            else:
                return render_template("results2.html", results=res)
    return "Image processing fail"


@app.route("/cnn.html", methods=['GET'])
def index():
    return render_template("cnn.html")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~KMEANS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


@app.route('/kmeansupload')
def upload_form():
    return render_template('kmeansUpload.html')


@app.route('/kmeansresult')
def kmeans_result():
    return render_template('kmeansResult.html')


@app.route('/kmeans.html')
def kmeans_upload_form():
    path = "static/kmeans"
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r.replace("static/", ""), file).replace("\\", "/"))
    print(files)
    return render_template('kmeans.html', files=files)


@app.route('/kmeans', methods=['POST'])
def kmeans_clear():
    if request.method == 'POST':
        os.system("ls static/kmeans/*.png")
    return redirect('/kmeans')


@app.route('/runkmeans', methods=['POST'])
def kmeans_run():
    count = 0
    for filename in os.listdir("static/kmeans/"):
        if filename.endswith(".png"):
            count = count + 1
            image = plt.imread("static/kmeans/" + filename)
            data = image.reshape(1, 32 * 32 * 3)
            if 'test' in locals():
                test = np.concatenate((test, data))
            else:
                test = data

    kmeans = KMeans(n_clusters=3, random_state=0).fit(test)
    class1 = []
    class2 = []
    class3 = []
    scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for filename in os.listdir("static/kmeans/"):
        if filename.endswith(".png"):
            image = plt.imread("static/kmeans/" + filename)
            data = image.reshape(1, 32 * 32 * 3)
            clusters = kmeans.predict(data)
            if (clusters[0] == 0):
                if (filename.startswith("dog")):
                    scores[0] += 1
                if (filename.startswith("auto")):
                    scores[1] += 1
                if (filename.startswith("ship")):
                    scores[2] += 1
                class1.append("kmeans/" + filename)
            if (clusters[0] == 1):
                if (filename.startswith("dog")):
                    scores[3] += 1
                if (filename.startswith("auto")):
                    scores[4] += 1
                if (filename.startswith("ship")):
                    scores[5] += 1
                class2.append("kmeans/" + filename)
            if (clusters[0] == 2):
                if (filename.startswith("dog")):
                    scores[6] += 1
                if (filename.startswith("auto")):
                    scores[7] += 1
                if (filename.startswith("ship")):
                    scores[8] += 1
                class3.append("kmeans/" + filename)
    flag = ""
    avg = (scores[0] + scores[1] + scores[2] + scores[3] + scores[4] + scores[5] + scores[6] + scores[7] + scores[
        8]) / 9 - 10
    if ((scores[0] + scores[1] + scores[2] > 300) and (scores[0] * scores[1] == 0) and (
            scores[3] + scores[4] + scores[5] > 300) and (scores[3] * scores[4] == 0) and (
            scores[6] + scores[7] + scores[8] > 300) and (scores[6] * scores[7] == 0)):
        flag = "tuskcon_flag{P-H4ck3d_your_w4y_to_100%}"
    elif (scores[0] > avg and scores[1] > avg and scores[2] > avg and scores[3] > avg and scores[4] > avg and scores[
        5] > avg and scores[6] > avg and scores[7] > avg and scores[8] > avg):
        flag = "tuskcon_flag{D4t4_15_b3tt3r_w1th_sup3rv1s1on}"
    return render_template('kmeansResult.html', class1=class1, class2=class2, class3=class3, scores=scores, flag=flag)


@app.route('/kmeansupload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        images = request.files.to_dict()  # convert multidict to dict
        for image in images:  # image will be the key
            file_name = secure_filename(images[image].filename)
            images[image].save(os.path.join('static/kmeans/', file_name))
            return redirect('/kmeans')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~POISON~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def Sentiment_process2(sent):
    return (sent)


def text_processing2(text):
    nopunc = []
    for char in text:
        nopunc.append(char)
    nopunc = (''.join(nopunc)).lower()
    return [word for word in nopunc.split()]


def preproc2(sample):
    with open("./static/poison/tokenizer", "rb") as file:
        tokenizer_data = file.read()
        tokenizer = pickle.loads(tokenizer_data)

    sample = text_processing2(sample)
    sample = tokenizer.texts_to_sequences(sample)
    simple_list = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(reverse_word_map)
    for sublist in sample:
        for item in sublist:
            simple_list.append(item)
    simple_list = [simple_list]
    maxlen = 50
    sample_review = sequence.pad_sequences(simple_list, padding='post', maxlen=maxlen)
    return sample_review


def calc2(word):
    clear_session()
    nn2 = tf.keras.models.load_model('./static/poison/poison.h5')
    # nn2.summary()
    # print(preproc(word))
    ans = nn2.predict(preproc2(word))
    return (ans)


@app.route('/poison.html', methods=['GET'])
def posion():
    # Add string to csv
    return render_template('poison.html')


@app.route('/addpoison', methods=['GET'])
def addpoison():
    # Add string to csv
    wordd = request.args.get("data")
    print(wordd)
    if wordd != None:
        with open("./static/poison/poison.csv", "a") as myfile:
            myfile.writelines(wordd + ",0\n")
    return render_template('addpoison.html')


@app.route('/rebuildpoison', methods=['GET'])
def rebuildpoison():
    # Rebuild model
    clear_session()
    df = pd.read_csv("./static/poison/poison.csv")

    df["Sentiment"] = Sentiment_process2(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing2)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    maxlen = 50
    # Making the train and test statements to be of size 50 by truncating or padding accordingly

    X_train = sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

    model = Sequential([Embedding(10000, 17),
                        GlobalAveragePooling1D(),
                        Dense(17, activation="relu"),
                        Dense(12, activation="relu"),
                        Dense(1, activation="sigmoid")])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy is : ", accuracy * 100)

    model.save('./static/poison/poison.h5')
    pickle_data = pickle.dumps(tokenizer)
    # storing the serialized output into a file in current directory
    with open("./static/poison/tokenizer", "wb") as file:
        file.write(pickle_data)

    return render_template('rebuildpoison.html')


@app.route('/checkpoison', methods=['GET'])
def checkpoison():
    # Check the value of string,
    df = pd.read_csv("./static/poison/poison.csv")

    df["Sentiment"] = Sentiment_process2(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing2)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(num_words=None, filters='', lower=True, split=' ', char_level=False, oov_token=None,
                          document_count=0)
    tokenizer.fit_on_texts(x)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    val = calc2("BEES")
    if val > 0.5:
        return render_template('poisonflag.html', flag="tuskcon_flag{P0150NED}")
    return render_template('checkpoison.html', score=val)


@app.route('/resetpoison', methods=['GET'])
def resetpoison():
    # Check the value of string,
    a = open('./static/poison/poison2.csv', 'r')
    read_content = a.read()
    b = open('./static/poison/poison.csv', 'w')
    b.write(read_content)

    return render_template('resetpoison.html')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~POISON 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def Sentiment_process(sent):
    return (sent)


def text_processing(text):
    nopunc = []
    for char in text:
        nopunc.append(char)
    nopunc = (''.join(nopunc)).lower()
    return [word for word in nopunc.split()]


def preproc5(sample):
    df = pd.read_csv("./static/poison2/poison.csv")
    df["Sentiment"] = Sentiment_process(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    sample = text_processing(sample)
    sample = tokenizer.texts_to_sequences(sample)
    simple_list = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(reverse_word_map)
    for sublist in sample:
        for item in sublist:
            simple_list.append(item)
    simple_list = [simple_list]
    maxlen = 50
    sample_review = sequence.pad_sequences(simple_list, padding='post', maxlen=maxlen)
    return sample_review


def calc5(word):
    clear_session()
    nn2 = tf.keras.models.load_model('./static/poison2/poison.h5')
    # nn2.summary()
    # print(preproc(word))
    ans = nn2.predict(preproc5(word))
    return (ans)


@app.route('/poison2.html', methods=['GET'])
def posion2():
    # Add string to csv
    return render_template('poison2.html')


@app.route('/addpoison2', methods=['GET'])
def addpoison2():
    # Add string to csv
    wordd = request.args.get("data")
    print(wordd)
    if wordd != None:
        with open("./static/poison2/poison.csv", "a") as myfile:
            myfile.writelines(re.sub('\W', '', wordd) + ",0\n")
    return render_template('addpoison2.html')


@app.route('/rebuildpoison2', methods=['GET'])
def rebuildpoison2():
    # Rebuild model
    clear_session()
    df = pd.read_csv("./static/poison2/poison.csv")

    df["Sentiment"] = Sentiment_process(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    maxlen = 50
    # Making the train and test statements to be of size 50 by truncating or padding accordingly

    X_train = sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

    model = Sequential([Embedding(10000, 17),
                        GlobalAveragePooling1D(),
                        Dense(17, activation="relu"),
                        Dense(12, activation="relu"),
                        Dense(1, activation="sigmoid")])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy is : ", accuracy * 100)

    model.save('./static/poison2/poison.h5')

    return render_template('rebuildpoison2.html')


@app.route('/checkpoison2', methods=['GET'])
def checkpoison2():
    # Check the value of string,
    df = pd.read_csv("./static/poison2/poison.csv")

    df["Sentiment"] = Sentiment_process(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(num_words=None, filters='', lower=True, split=' ', char_level=False, oov_token=None,
                          document_count=0)
    tokenizer.fit_on_texts(x)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    val = calc5("BEES")
    if val > 0.5:
        return render_template('poisonflag2.html', flag="tuskcon_flag{P0150NED23L3CTR1CB00G4L00}")
    return render_template('checkpoison2.html', score=val)


@app.route('/resetpoison2', methods=['GET'])
def resetpoison2():
    # Check the value of string,
    a = open('./static/poison2/poison2.csv', 'r')
    read_content = a.read()
    b = open('./static/poison2/poison.csv', 'w')
    b.write(read_content)

    return render_template('resetpoison.html')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TOKEN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def Sentiment_processToken(sent):
    return (sent)


def text_processingToken(text):
    nopunc = []
    for char in text:
        nopunc.append(char)
    nopunc = (''.join(nopunc)).lower()
    return [word for word in nopunc.split()]


def calcToken(word, ctr1, ctr2):
    nn2 = tf.keras.models.load_model('./static/token/model.h5')
    print(type(nn2))
    # nn2.summary()
    # print(preproc(word))
    ans = nn2.predict(preprocToken(word, ctr1, ctr2))
    clear_session()
    return (ans)


def preprocToken(sample, ctr1, ctr2):
    a = open('./static/token/test.csv', 'r')
    read_content = a.readlines()
    a.close()
    count = 0
    # Strips the newline character
    b = open('./static/token/temp.csv', 'w')
    for line in read_content:
        count += 1
        if (count == int(ctr1) or count == int(ctr2)):
            print("Line{}: {}".format(count, line.strip()))
        else:
            b.write(line)
    b.close()
    df = pd.read_csv("./static/token/temp.csv")
    df["Sentiment"] = Sentiment_processToken(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processingToken)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(int)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    sample = text_processingToken(sample)
    sample = tokenizer.texts_to_sequences(sample)
    simple_list = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(reverse_word_map)
    for sublist in sample:
        for item in sublist:
            simple_list.append(item)
    simple_list = [simple_list]
    maxlen = 50
    sample_review = sequence.pad_sequences(simple_list, padding='post', maxlen=maxlen)
    return sample_review


@app.route('/checktoken', methods=['GET'])
def checktoken():
    ctr1 = request.args.get("line1")
    if ctr1 is None:
        return render_template('token.html')
    ctr2 = request.args.get("line2")
    if ctr2 is None:
        return render_template('token.html')
    val = calcToken("SECRETKEY", ctr1, ctr2)
    print(val)
    if val > 0.8:
        return render_template('tokenflag.html', flag="tuskcon_flag{T0K3N1Z3M3}")
    return render_template('checktoken.html', score=val)


@app.route("/token.html", methods=['GET'])
def token():
    return render_template("token.html")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~WAF~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def text_processing(text):
    nopunc = []
    for char in text:
        nopunc.append(char)
    nopunc = (''.join(nopunc))
    return [word for word in nopunc.split()]


def preproc(sample):
    df = pd.read_csv("./static/wafdatasiudhasiudhsiuhauiashsui/waf.csv")
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=False, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    sample = text_processing(sample)
    sample = tokenizer.texts_to_sequences(sample)
    simple_list = []
    for sublist in sample:
        for item in sublist:
            simple_list.append(item)
    simple_list = [simple_list]
    maxlen = 50
    sample_review = sequence.pad_sequences(simple_list, padding='post', maxlen=maxlen)
    return sample_review


def calc(word, nn2):
    ans = nn2.predict(preproc(word))
    return (ans)


@app.route('/waf.html', methods=['GET'])
def waf():
    # Check the value of string,
    df = pd.read_csv("./static/wafdatasiudhasiudhsiuhauiashsui/waf.csv")

    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]

    tokenizer = Tokenizer(num_words=None, filters='', lower=True, split=' ', char_level=False, oov_token=None,
                          document_count=0)
    tokenizer.fit_on_texts(x)
    a = request.full_path
    clear_session()
    nn2 = tf.keras.models.load_model('./static/wafdatasiudhasiudhsiuhauiashsui/waf.h5')
    for i in range(5, len(a), 5):
        # print(a[i:i+5])
        val = calc(a[i:i + 5], nn2)
        if val < 0.15:
            return render_template('0day.html')
    wordd = request.args.get("addenv")
    if wordd is None:
        return render_template('waf.html')
    try:
        wordd = (base64.b64decode(wordd)).decode("utf-8")
    except:
        return render_template('b64error.html')
    if "() { :; };" in wordd:
        os.system(wordd.replace("() { :; };", ""))
    return render_template('waf.html')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~THEFT 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def get_predictions2(image):
    # open and read the file after the appending:
    clear_session()
    with open("./static/cnn/picklemodel", "rb") as file:
        pickle_data = file.read()
        model = pickle.loads(pickle_data)
    preds = model.predict(image)
    # print(preds.argmax(axis=-1))
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


@app.route("/theft1_predict", methods=["POST"])
def theft1_predict():
    base_img = Image.open("./static/owl.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("loggerhead"):

            loggerhead_img = Image.open(io.BytesIO(request.files["loggerhead"].read()))
            s1 = str(phash(loggerhead_img))
            s2 = str(phash(base_img))

            loggerhead_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            loggerhead_mat = prepare_image(loggerhead_img)

            # read the image in PIL format
            loggerhead_label, loggerhead_conf, top_preds = get_predictions2(loggerhead_mat)

            res = {}
            res["a"] = "loggerhead" in loggerhead_label
            res["b"] = loggerhead_conf
            res["c"] = loggerhead_label
            res["d"] = loggerhead_dist
            res["e"] = top_preds
            res["backpage"] = "./theft1.html"

            # if "great_grey_owl" in loggerhead_label and loggerhead_conf >= THRESHOLD and loggerhead_dist <= PHASH_TRESH:
            if "loggerhead" in loggerhead_label and loggerhead_conf >= THRESHOLD and loggerhead_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{ST34LYST34Y}", results=res)
            else:
                return render_template("results.html", results=res)

    return "Image processing fail"


@app.route("/theft1.html", methods=['GET'])
def theft1():
    return render_template("theft1.html")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~THEFT 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def get_predictions3(image):
    # open and read the file after the appending:

    key = b"alejandroalejand"
    nonce = b"alejandroalejand"
    f = open("./static/cnn/encpickle", "rb")
    daytas = f.read()

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt(daytas)
    clear_session()
    model = pickle.loads(plaintext)
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


@app.route("/theft2_predict", methods=["POST"])
def theft2_predict():
    base_img = Image.open("./static/turtle.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("turtle"):

            great_grey_owl_img = Image.open(io.BytesIO(request.files["turtle"].read()))
            s1 = str(phash(great_grey_owl_img))
            s2 = str(phash(base_img))

            great_grey_owl_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            great_grey_owl_mat = prepare_image(great_grey_owl_img)

            # read the image in PIL format
            great_grey_owl_label, great_grey_owl_conf, top_preds = get_predictions3(great_grey_owl_mat)

            res = {}
            res["a"] = "great_grey_owl" in great_grey_owl_label
            res["b"] = great_grey_owl_conf
            res["c"] = great_grey_owl_label
            res["d"] = great_grey_owl_dist
            res["e"] = top_preds
            res["backpage"] = "./theft2.html"

            # if "loggerhead" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
            if "great_grey_owl" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{D1D3NCRYPT10NH3LP?}", results=res)
            else:
                return render_template("results.html", results=res)

    return "Image processing fail"


@app.route("/theft2.html", methods=['GET'])
def theft2():
    return render_template("theft2.html")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~THEFT 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def get_predictions4(image):
    # open and read the file after the appending:

    # print(type(uasyhdiuhasdiuhadsimodel))
    key = b"alejandroalejand"
    nonce = b"alejandroalejand"
    f = open("./static/cnn/encpickle", "rb")
    daytas = f.read()

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt(daytas)
    clear_session()
    model = pickle.loads(plaintext)
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


@app.route("/theft3_predict", methods=["POST"])
def theft3_predict():
    base_img = Image.open("./static/iguana.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("iguana"):

            iguana_img = Image.open(io.BytesIO(request.files["iguana"].read()))
            s1 = str(phash(iguana_img))
            s2 = str(phash(base_img))

            iguana_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            iguana_mat = prepare_image(iguana_img)

            # read the image in PIL format
            iguana_label, iguana_conf, top_preds = get_predictions4(iguana_mat)

            res = {}
            res["a"] = "bee" in iguana_label
            res["b"] = iguana_conf
            res["c"] = iguana_label
            res["d"] = iguana_dist
            res["e"] = top_preds
            res["backpage"] = "./theft3.html"

            # if "common_iguana" in iguana_label and iguana_conf >= THRESHOLD and iguana_dist <= PHASH_TRESH:
            if "bee" in iguana_label and iguana_conf >= THRESHOLD and iguana_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{D0NT@M31ML4ZY}", results=res)
            else:
                return render_template("results.html", results=res)

    return "Image processing fail"


@app.route("/theft3.html", methods=['GET'])
def theft3():
    return render_template("theft3.html")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SALT Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_predictionsSalt(image):
    # open and read the file after the appending:
    clear_session()
    with open("./static/cnn/picklemodel", "rb") as file:
        pickle_data = file.read()
        model = pickle.loads(pickle_data)

    preds = model.predict(image)
    # print(preds.argmax(axis=-1))
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds

def prepare_imageSalt(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(IMAGE_DIMS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SALT 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@app.route("/salt1_predict", methods=["POST"])
def salt1_predict():
    base_img = Image.open("./static/turtle.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("turtle"):

            great_grey_owl_img = Image.open(io.BytesIO(request.files["turtle"].read()))
            s1 = str(phash(great_grey_owl_img))
            s2 = str(phash(base_img))

            great_grey_owl_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            great_grey_owl_mat = prepare_imageSalt(great_grey_owl_img)

            noise = Image.open('./static/cnn/salt.png')
            noise2 = prepare_imageSalt(noise)

            imagemod = great_grey_owl_mat + noise2
            # read the image in PIL format
            great_grey_owl_label, great_grey_owl_conf, top_preds = get_predictionsSalt(preprocess_input(imagemod))

            res = {}
            res["a"] = "great_grey_owl" in great_grey_owl_label
            res["b"] = great_grey_owl_conf
            res["c"] = great_grey_owl_label
            res["d"] = great_grey_owl_dist
            res["e"] = top_preds
            res["backpage"] = "./salt1.html"

            # if "loggerhead" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
            if "great_grey_owl" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{S4LTS4R3FUN}", results=res)
            else:
                return render_template("results.html", results=res)
    return "Image processing fail"


@app.route("/salt1.html", methods=['GET'])
def salt1():
    return render_template("salt1.html")

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SALT 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""



@app.route("/salt2_predict", methods=["POST"])
def salt2_predict():
    base_img = Image.open("./static/turtle.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("turtle"):

            great_grey_owl_img = Image.open(io.BytesIO(request.files["turtle"].read()))
            s1 = str(phash(great_grey_owl_img))
            s2 = str(phash(base_img))

            great_grey_owl_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            imagemod = great_grey_owl_img.filter(ImageFilter.GaussianBlur(0.6))
            # read the image in PIL format
            imagemod = prepare_imageSalt(imagemod)
            great_grey_owl_label, great_grey_owl_conf, top_preds = get_predictionsSalt(preprocess_input(imagemod))
            print(top_preds)

            res = {}
            res["a"] = "great_grey_owl" in great_grey_owl_label
            res["b"] = great_grey_owl_conf
            res["c"] = great_grey_owl_label
            res["d"] = great_grey_owl_dist
            res["e"] = top_preds
            res["backpage"] = "./salt2.html"

            # if "loggerhead" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
            if "great_grey_owl" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{KN0WNS4LTS4R3B4D}", results=res)
            else:
                return render_template("results.html", results=res)
    return "Image processing fail"

@app.route("/salt2.html", methods=['GET'])
def salt2():
    return render_template("salt2.html")

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SALT 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


@app.route("/salt3_predict", methods=["POST"])
def salt3_predict():
    base_img = Image.open("./static/turtle.jpg").resize(IMAGE_DIMS)
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("turtle"):
            great_grey_owl_img = Image.open(io.BytesIO(request.files["turtle"].read()))
            s1 = str(phash(great_grey_owl_img))
            s2 = str(phash(base_img))
            great_grey_owl_dist = sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))
            loggerhead_mat =  prepare_imageSalt(great_grey_owl_img)
            great_grey_owl_label = []
            great_grey_owl_conf = []
            top_preds = []
            for i in range(5):
                noise = np.random.normal(0, 5, (1, 224, 224, 3))
                imagemod = loggerhead_mat[0] + noise
                # read the image in PIL format
                a, b, c = get_predictionsSalt(preprocess_input(imagemod))
                great_grey_owl_label.append(a)
                great_grey_owl_conf.append(b)
                top_preds.append(c)
            res = {}
            res["a"] = "great_grey_owl" in max(set(great_grey_owl_label), key = great_grey_owl_label.count)
            res["b"] = great_grey_owl_conf
            res["c"] = great_grey_owl_label
            res["d"] = great_grey_owl_dist
            res["e"] = top_preds
            res["backpage"] = "./salt3.html"
            # if "loggerhead" in great_grey_owl_label and great_grey_owl_conf >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
            if "great_grey_owl" in max(set(great_grey_owl_label), key = great_grey_owl_label.count) and sum(great_grey_owl_conf)/5 >= THRESHOLD and great_grey_owl_dist <= PHASH_TRESH:
                return render_template("win.html", flag="tuskcon_flag{L1LS4LT<B1GS4LT}", results=res)
            else:
                return render_template("results.html", results=res)
    return "Image processing fail"


@app.route("/salt3.html", methods=['GET'])
def salt3():
    return render_template("salt3.html")

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Pickle 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def get_predictions5(image, picklemodel):
    # open and read the file after the appending:
    clear_session()
    model = pickle.loads(picklemodel)
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


@app.route("/pickle1_predict", methods=["POST"])
def pickle1_predict():
    base_img = Image.open("./static/bee.jpg").resize(IMAGE_DIMS)

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("model"):
            picklemodel = request.files["model"].read()
            bee_mat = prepare_image(base_img)

            # read the image in PIL format
            bee_label, bee_conf, top_preds = get_predictions5(bee_mat, picklemodel)

            res = {}
            res["bee"] = "bee" in bee_label
            res["is_class"] = bee_conf
            res["conf"] = bee_label
            res["top_5"] = top_preds
            res["backpage"] = "./pickle1.html"
            return render_template("pickle1results.html", results=res)

    return "Model processing fail"


@app.route("/pickle1.html", methods=['GET'])
def pickle1():
    return render_template("pickle1.html")


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SHIFT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def preprocShift(sample):
    df = pd.read_csv("./static/shift/shift.csv")
    df["Sentiment"] = Sentiment_process(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(float)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    with open("./static/shift/tokenizer", "rb") as file:
        tokenizer_data = file.read()
        tokenizer = pickle.loads(tokenizer_data)

    sample = text_processing(sample)
    sample = tokenizer.texts_to_sequences(sample)
    simple_list = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(reverse_word_map)
    for sublist in sample:
        for item in sublist:
            simple_list.append(item)
    simple_list = [simple_list]
    maxlen = 50
    sample_review = sequence.pad_sequences(simple_list, padding='post', maxlen=maxlen)
    return sample_review


def calcShift(word):
    clear_session()
    nn2 = tf.keras.models.load_model('./static/shift/shift.h5')
    # nn2.summary()
    # print(preproc(word))
    ans = nn2.predict(preprocShift(word))
    return (ans)


@app.route('/shift.html', methods=['GET'])
def shift():
    # Add string to csv
    return render_template('shift.html')


@app.route('/addshift', methods=['GET'])
def addshift():
    # Add string to csv
    val = calcShift("siudhsi")
    with open("./static/shift/shift.csv", "a") as myfile:
        myfile.writelines("siudhsi," + str(val[0][0]) + "\n")
    return render_template('addshift.html')


@app.route('/rebuildshift', methods=['GET'])
def rebuildshift():
    # Rebuild model
    clear_session()
    df = pd.read_csv("./static/shift/shift.csv")

    df["Sentiment"] = Sentiment_process(list(df['Sentiment']))
    df["Review"] = df['Review']
    df["BagOfWords"] = df["Review"].apply(text_processing)
    x = df["BagOfWords"]
    df["Sentiment"] = df["Sentiment"].astype(str).astype(float)
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    tokenizer = Tokenizer(
        num_words=None,
        filters='',
        lower=True, split=' ', char_level=False, oov_token=None,
        document_count=0,
    )
    tokenizer.fit_on_texts(x)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    maxlen = 50
    # Making the train and test statements to be of size 50 by truncating or padding accordingly

    X_train = sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print(reverse_word_map)
    model = Sequential([Embedding(10000, 17),
                        GlobalAveragePooling1D(),
                        Dense(17, activation="relu"),
                        Dense(12, activation="relu"),
                        Dense(1, activation="sigmoid")])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy is : ", accuracy * 100)

    model.save('./static/shift/shift.h5')
    pickle_data = pickle.dumps(tokenizer)
    # storing the serialized output into a file in current directory
    with open("./static/shift/tokenizer", "wb") as file:
        file.write(pickle_data)

    return render_template('rebuildshift.html')


@app.route('/checkshift', methods=['GET'])
def checkshift():
    # Check the value of string,
    val = calcShift("siudhsi")
    print(val)
    if val > 0.80:
        return render_template('shiftflag2.html', flag="tuskcon_flag{G3TSH1FTY}")
    return render_template('checkshift.html', score=val)


@app.route('/resetshift', methods=['GET'])
def resetshift():
    # Check the value of string,
    a = open('./static/shift/shift2.csv', 'r')
    read_content = a.read()
    b = open('./static/shift/shift.csv', 'w')
    b.write(read_content)
    return render_template('resetshift.html')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


@app.route('/')
def main_page():
    return render_template('index.html')


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == "__main__":
    # This section of code is here to embed forensics in the running file and should be ignored for challenges other than theft3 :)
    asihduihsuihasduihasd = "asiuhasdihaiuasdhiasuhiasdusi"
    key = b"alejandroalejand"
    nonce = b"alejandroalejand"
    f = open("./static/cnn/encpickle", "rb")
    daytas = f.read()

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    asiudhasiuhasiuhasdiuplaintext = cipher.decrypt(daytas)
    uasyhdiuhasdiuhadsimodel = pickle.loads(asiudhasiuhasiuhasdiuplaintext)
    # END section for theft3 :)
    app.run(host='0.0.0.0')
