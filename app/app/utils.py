import importlib
from keras.models import Model
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

class ImagePretrainedModel:
    def __init__(self, model_zoo, model_name, weights='imagenet', include_top=True):
        # Programmatically import relevant model
        module = importlib.import_module(model_zoo["keras_model_zoo"][model_name]["pkg"])
        class_ = getattr(module, model_name)
        base_model = class_(weights=weights, include_top=include_top)
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        self.graph = tf.get_default_graph()
        self.image_size = eval(model_zoo["keras_model_zoo"][model_name]["image_size"])[:2]
        self.preprocess_input_func = getattr(module, "preprocess_input")

    def get_embbeding_from_bytes(self, img_bytes, normalize=True):
        img = Image.open(img_bytes.file) # img is from PIL.Image.open(path)
        img = img.convert('RGB') # Make sure img is color
        img = img.resize(self.image_size)  # VGG must take a 224x224 img as an input
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = self.preprocess_input_func(x)  # Subtracting avg values for each pixel
        with self.graph.as_default():
            feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
            if normalize:
                result = feature / np.linalg.norm(feature) # Normalize
            else:
                result = feature
            return result.tolist()

    def get_embbeding_from_bytes_cv2(self, img_bytes, normalize=True):
        # use numpy to construct an array from the bytes
        img = np.asarray(bytearray(img_bytes.file.read()), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, self.image_size)  # VGG must take a 224x224 img as an input
        x = np.asarray(img, dtype="float32")  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = self.preprocess_input_func(x)  # Subtracting avg values for each pixel
        with self.graph.as_default():
            feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
            if normalize:
                result = feature / np.linalg.norm(feature)  # Normalize
            else:
                result = feature
            return result.tolist()

    def get_embbeding_from_bytes_list(self, fileList, normalize=True):
        img_list = []
        for img_bytes in fileList:
            img = Image.open(img_bytes.file) # img is from PIL.Image.open(path)
            img = img.convert('RGB') # Make sure img is color
            img = img.resize(self.image_size)  # VGG must take a 224x224 img as an input
            x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
            img_list.append(x)

        x = np.stack(img_list, axis=0) # n_img * (H, W, C)->(n_img, H, W, C), where the first elem is the number of img
        x = self.preprocess_input_func(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x)  # (1, 4096) -> (4096, )
            if normalize:
                result = feature / np.linalg.norm(feature) # Normalize
            else:
                result = feature
            return result.tolist()

    def get_embbeding_from_bytes_list_cv2(self, fileList, normalize=True):
        img_list = []
        for img_bytes in fileList:
            img = np.frombuffer(bytearray(img_bytes.file.read()), dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, self.image_size)  # VGG must take a 224x224 img as an input
            img_list.append(np.asarray(img, dtype="float32")) # To np.array. Height x Width x Channel. dtype=float32

        x = np.stack(img_list, axis=0)  # n_img * (H, W, C)->(n_img, H, W, C), where the first elem is the number of img
        x = self.preprocess_input_func(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x)  # (1, 4096) -> (4096, )
            if normalize:
                result = feature / np.linalg.norm(feature)  # Normalize
            else:
                result = feature
            return result.tolist()
