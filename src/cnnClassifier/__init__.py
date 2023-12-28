import os
import sys
import logging

"""
%(asctime)s: store the current timestamp while running the project
%(levelname)s: what type of logs we want to store . Here we wanna store information level log
%(module)s: which module is running the file
%(message)s: will display the message if there are any errors
"""

logging_str = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"

# Creating a log folder and inside it a running_logs.log file. Inside this file all the logs will be saved
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("cnnClassifierLogger")
