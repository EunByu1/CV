from tensorflow.keras.models import Sequntial
from tensorflow.keras.layers import Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

data_path = pathlib.Path("dataets/stanford_dogs/images/images")

train_ds = image_dataset_from_directory(data_path, validation_split=0.2, subset="training", seed=123, image_size=(224,224), batch_size=16)
test_ds  = image_dataset_from_directory(data_path, validation_split=0.2, subset="validation", seed=123, image_size=(224,224), batch_size=16)

base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

cnn.Sequential()
cnn.add(Rescaling())