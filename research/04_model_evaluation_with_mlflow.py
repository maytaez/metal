#%% 
import os
from typing import Any
%pwd
# %%
os.chdir("../")
# %% collect mlflow url
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/maytaez/metal.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="maytaez"
os.environ["MLFLOW_TRACKING_PASSWORD"]="189dd8fc849777ce77eee82487a289a1937873ea"

# %%
import tensorflow as tf
model=tf.keras.models.load_model("artifacts/training/model.h5")

#update the entity
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_url: str
    params_image_size: list
    params_batch_size: int

from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories,save_json

#update configuration manager
class ConfigurationManageer:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)
            create_directories([self.config.artifacts_root])

    def get_evaluation_config(self)-> EvaluationConfig:
         eval_config=EvaluationConfig(
              path_of_model="artifacts/training/model.h5",
              training_data="artifacts/data_ingestion/casting_data",
              mlflow_url="https://dagshub.com/maytaez/metal.mlflow",
              all_params=self.params,
              params_image_size=self.params.IMAGE_SIZE,
              params_batch_size=self.params.BATCH_SIZE
         )
         return eval_config
# %% updating the components
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse

# %% update the configuration manager

# class ConfigurationManager:
#      def __init__(
#         self,
#         config_filepath=CONFIG_FILE_PATH,
#         params_filepath=PARAMS_FILE_PATH):
#         self.config=read_yaml(config_filepath)
#         self.params=read_yaml(params_filepath)

#      def get_evaluation_config(self)-> EvaluationConfig:
#           eval_config=EvaluationConfig(
#                path_of_model="artifacts/training/model.h5",
#                training_data="artifacts/data_ingestion/cating_data",
#                mlflow_url="https://dagshub.com/maytaez/metal.mlflow",
#                all_params=self.params,
#                params_image_size=self.params.IMAGE_SIZE,
#                params_batch_size=self.params.BATCH_SIZE
#           ) 
#           return eval_config
          

#%%
class Evaluation:
     def __init__(self,config: EvaluationConfig):
          self.config=config

     def _valid_generator(self):# to prepare test set
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)

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

     @staticmethod
     def load_model(path:Path)->tf.keras.Model:
            return tf.keras.models.load_model(path)
    
     def evaluation(self):
            self.model=self.load_model(self.config.path_of_model)
            self._valid_generator()
            self.score=model.evaluate(self.valid_generator)
            self.save_score()
            
     def save_score(self):
          scores={"loss":self.score[0],"accuracy":self.score[1]}
          save_json(path=Path("scores.json"),data=scores)
     def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

        
        

# %%
try:
    config = ConfigurationManageer()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()

except Exception as e:
   raise e
# %%
