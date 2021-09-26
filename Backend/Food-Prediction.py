import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

K.clear_session()
model_best = load_model('Model/best_model_3class.hdf5', compile=False)
food_list = ['donuts', 'pizza', 'samosa']


def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()


def main():
    images = []
    samos = "Sample/Homemade Samosa.jpg"
    # images.append('donut.jpg')
    # images.append('pizza.jpg')
    images.append(samos)
    predict_class(model_best, images, True)


if __name__ == "__main__":
    main()
