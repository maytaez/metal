from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrasiningPipeline,
)
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from src.cnnClassifier.pipeline.stage_03_model_trainer import (
    ModelTrainingPipeline,
)


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>> stage {STAGE_NAME} started <<< ")
    obj = DataIngestionTrasiningPipeline()
    obj.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>> stage {STAGE_NAME} started <<< ")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<\n\nx=========x")

except Exception as e:
    logger.exception(e)

STAGE_NAME = "Training"

try:
    logger.info(f">>> stage {STAGE_NAME} started <<< ")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<\n\nx=========x")

except Exception as e:
    logger.exception(e)
