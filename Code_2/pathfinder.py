import os
import keras
from keras.layers import Embedding
module_path = os.path.abspath(keras.__file__)
print("Module path:", module_path)
