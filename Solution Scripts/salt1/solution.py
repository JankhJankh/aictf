IMAGE_DIMS = (224,224)

from PIL import Image
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
import scipy
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

IMAGE_DIMS = (224,224)

def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(IMAGE_DIMS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image


noise = Image.open('./salt.png')
noise2 = prepare_image(noise)
image = Image.open('./turtle_owl.png')
image2 = prepare_image(image)
imod = image2 - noise2
scipy.misc.toimage(imod[0],cmin=0, cmax=255).save('./salted.png')

salt = Image.open('./salted.png')
salt2 = prepare_image(salt)
noise = Image.open('./salt.png')
noise2 = prepare_image(noise)
imod2 = salt2 + noise2
scipy.misc.toimage(imod2[0]).save('./returned.png')
