#%%
import os
from typing import Any
# %%
%pwd
# %%
#Update parameters(param.yaml) : as we are initalizing our model

#%%
#Creating the entity
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True) #frozen=True means we can't add anything to this class now apart from the below written code.
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

#%%
#update the configuration manager on src config
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            params_filepath= PARAMS_FILE_PATH):

            self.config= read_yaml(config_filepath)
            self.params= read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
         config=self.config.prepare_base_model

         create_directories([config.root_dir])

         prepare_base_model_config = PrepareBaseModelConfig(
              root_dir=Path(config.root_dir),
              base_model_path=Path(config.base_model_path),
              updated_base_model_path=Path(config.updated_base_model_path),
              params_image_size=self.params.IMAGE_SIZE,
              params_learning_rate=self.params.LEARNING_RATE,
              params_include_top=self.params.INCLUDE_TOP,
              params_weights=self.params.WEIGHTS,
              params_classes=self.params.CLASSES
            

         )
         return prepare_base_model_config
    

# %%
#Updating component
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

#this class will take PrepareBaseModelConfig configuration from above as defined
class PrepareBaseModel:
     def __init__(self,config: PrepareBaseModelConfig):
          self.config = config
    #defining methods
    #downloading VGG-16
     def get_base_model(self):
          self.model=tf.keras.applications.vgg16.VGG16(
               input_shape=self.config.params_image_size,
               weights=self.config.params_weights,
               include_top=self.config.params_include_top
          )

          self.save_model(path=self.config.base_model_path, model=self.model)

    #fine-tuning by adding custom layer according to the number of output classes
     @staticmethod
     def _prepare_full_model(model,classes,freeze_all,freeze_till,learning_rate):
          if freeze_all:
               for layer in model.layers:
                    model.trainable=False
          elif (freeze_till is not None) and (freeze_till>0):
               model.trainable=False

          flatten_in=tf.keras.layers.Flatten()(model.output)
          prediction=tf.keras.layers.Dense(
               units=classes,
               activation="softmax"
          )(flatten_in)

          full_model=tf.keras.models.Model(
               inputs=model.input,
               outputs=prediction
          )

          full_model.compile(
               optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
               loss=tf.keras.losses.CategoricalCrossentropy(),
               metrics=["accuracy"]
          )

          full_model.summary()
          return full_model
     
     #this full model also needs to be saved
     def update_base_model(self):
          self.full_model=self._prepare_full_model(
               model=self.model,
               classes=self.config.params_classes,
               freeze_all=True,
               freeze_till=None,
               learning_rate=self.config.params_learning_rate

          )

          self.save_model(path=self.config.updated_base_model_path,model=self.full_model)
    
    
    
    #defining save model function which will be static
     @staticmethod
     def save_model(path: Path, model: tf.keras.Model):
          model.save(path)
     

# %%
#Update the pipeline
try:
     config=ConfigurationManager()
     prepare_base_model_config=config.get_prepare_base_model_config()
     prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
     prepare_base_model.get_base_model()
     prepare_base_model.update_base_model()

except Exception as e:
     raise e
# %%
#base_model.h5 is the downloaded VGG-16 model
#updated_base_model.h5 will be used while training