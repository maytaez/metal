# metal

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional for database stuff]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py file
9. Update the dvc.yaml file
10. Update app.py

# How to run?

### STEPS:

Clone the repository

```bash
https://github.com/maytaez/metal
```

### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n metal python=3.8 -y
```

```bash
conda activate metal
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/maytaez/metal.mlflow \
MLFLOW_TRACKING_USERNAME=maytaez \
MLFLOW_TRACKING_PASSWORD=189dd8fc849777ce77eee82487a289a1937873ea \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/maytaez/metal.mlflow 

export MLFLOW_TRACKING_USERNAME=maytaez 
export MLFLOW_TRACKING_PASSWORD=189dd8fc849777ce77eee82487a289a1937873ea 