#Code yoinked from here: https://tcode2k16.github.io/blog/posts/picoctf-2018-writeup/general-skills/#solution-20


from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
from imagehash import phash
import numpy as np
from keras import backend as K

IMAGE_DIMS = (224, 224)
TREE_FROG_IDX = 31
TREE_FROG_STR = "tree_frog"
TURTLE_IDX = 33
TURTLE_STR = "loggerhead"
OWL_IDX = 24
OWL_STR = "great_grey_owl"
BEE_IDX = 309
BEE_STR = "bee"
# I'm pretty sure I borrowed this function from somewhere, but cannot remember
# the source to cite them properly.
def hash_hamming_distance(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))


def is_similar_img(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    dist = hash_hamming_distance(phash(image1), phash(image2))
    return dist <= 1


def prepare_image(image, target=IMAGE_DIMS):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    # return the processed image
    return image


def create_img(img_path, img_res_path, model_path, target_str, target_idx, des_conf=0.95):
    original_image = Image.open(img_path).resize(IMAGE_DIMS)
    original_image = prepare_image(original_image)
    model = load_model(model_path)

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01

    # Create a copy of the input image to hack on
    hacked_image = np.copy(original_image)

    # How much to update the hacked image in each iteration
    learning_rate = 0.01

    # Define the cost function.
    # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
    cost_function = model_output_layer[0, target_idx]

    # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
    # In this case, referring to "model_input_layer" will give us back image we are hacking.
    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    # Create a Keras function that we can call to calculate the current cost and gradient
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0

    # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
    # until it gets to at least 80% confidence
    while cost < 0.99:
        # Check how close the image is to our target class and grab the gradients we
        # can use to push it one more step in that direction.
        # Note: It's really important to pass in '0' for the Keras learning mode here!
        # Keras layers behave differently in prediction vs. train modes!
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        # Move the hacked image one step further towards fooling the model
        # print gradients
        hacked_image += np.sign(gradients) * learning_rate

        # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        print("Model's predicted likelihood that the image is a "+target_str+": {:.8}%".format(cost * 100))

    hacked_image = hacked_image.reshape((224,224,3))
    img = array_to_img(hacked_image)
    img.save(img_res_path)


# For some reason the result file neeeds to be png to work. :)
if __name__ == "__main__":
    create_img("./trixi.png", "./trixi_frog.png", "./model.h5", TREE_FROG_STR, TREE_FROG_IDX)
    assert is_similar_img("./trixi_frog.png", "./trixi_frog.png")
    create_img("./owl.jpg", "./owl_turtle.png", "./model.h5", TURTLE_STR, TURTLE_IDX)
    assert is_similar_img("./owl.jpg", "./owl_turtle.png")
    create_img("./turtle.jpg", "./turtle_owl.png", "./model.h5", OWL_STR, OWL_IDX)
    assert is_similar_img("./turtle.jpg", "./turtle_owl.png")
    create_img("./iguana.jpg", "./iguana_bee.png", "./model.h5", BEE_STR, BEE_IDX)
    assert is_similar_img("./iguana.jpg", "./iguana_bee.png")
