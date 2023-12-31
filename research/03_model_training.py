# # %%
# import os
# #%%
# %pwd
# os.chdir("../")

# # update config.yaml file

# # %%Update entity
# from dataclasses import dataclass
# from pathlib import Path


# @dataclass(frozen=True)
# class TrainingConfig:
#     root_dir: Path
#     trained_model_path: Path
#     updated_base_model_path: Path  # fine-tuned model
#     training_data: Path
#     params_epochs: int
#     params_batch_size: int
#     params_is_augmentation: bool
#     params_image_size: list


# # %%Update Configuration Manager
# from src.cnnClassifier.constants import *
# from src.cnnClassifier.utils.common import read_yaml, create_directories
# import tensorflow as tf
# # from keras.preprocessing.image import load_img


# class ConfigurationManager:
#     def __init__(
#         self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
#     ):
#         self.config = read_yaml(config_file_path)
#         self.params = read_yaml(params_file_path)

#         create_directories([self.config.artifacts_root])

#     # model training configuration
#     def get_training_config(self) -> TrainingConfig:
#         training = self.config.training
#         prepare_base_model = self.config.prepare_base_model
#         params = self.params
#         training_data = os.path.join(
#             self.config.data_ingestion.unzip_dir, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
#         )
#         create_directories([Path(training.root_dir)])

#         training_config = TrainingConfig(
#             root_dir=Path(training.root_dir),
#             trained_model_path=Path(training.trained_model_path),
#             updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
#             training_data=Path(training_data),
#             params_epochs=params.EPOCHS,
#             params_batch_size=params.BATCH_SIZE,
#             params_is_augmentation=params.AUGMENTATION,
#             params_image_size=params.IMAGE_SIZE,
#         )
#         return training_config


# # %%Update component

# import os
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time

# # initializing training


# class Training:
#     def __init__(self, config:TrainingConfig):
#         self.config = config

#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

#     # splitting train and valid set
#     def train_valid_generator(self):
#         datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear",
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

#     # after training, saving the model
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)

#     # creating training function
#     def train(self):  # , callback_list: list):
#         self.steps_per_epoch = (
#             self.train_generator.samples // self.train_generator.batch_size
#         )
#         self.validation_steps = (
#             self.valid_generator.samples // self.valid_generator.batch_size
#         )

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_steps=self.validation_steps,
#             validation_data=self.valid_generator,
#             # callbacks=callback_list,
#         )

#         self.save_model(path=self.config.trained_model_path, model=self.model)


# #%% Updating the pipeline
# try:
#     config = ConfigurationManager()
#     training_config = config.get_training_config()
#     training = Training(config=training_config)
#     training.get_base_model()
#     training.train_valid_generator()
#     training.train()  # callbacks=callback_list)
# except Exception as e:
#     raise e

# # %%
# %%
import os
#%%
%pwd

#%%
# os.chdir("../")
# %pwd
# 'd:\\Bappy\\YouTube\\Kidney-Disease-Classification-Deep-Learning-Project'
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list


from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "kidney-ct-scan-image",
        )
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config


import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from PIL import Image


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)


try:
    config = ConfigurationManager()
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train()

except Exception as e:
    raise e

# %%
try:
    from PIL import Image
    print("Pillow is installed and working.")
except ImportError as e:
    print("Pillow is not installed or there is an import error:", e)
# %%
