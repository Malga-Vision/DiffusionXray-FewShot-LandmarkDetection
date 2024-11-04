#!/bin/bash

# Path to the config.json file
DDPM_CONFIG_PATH="ddpm_pretraining/config/config.json"
SSL_CONFIG_PATH="ssl_pretraining/config/config.json"

# Path to the main.py script
DDPM_PY_PATH="ddpm_pretraining/main.py"
SSL_PY_PATH="ssl_pretraining/main.py"

# Temporary file for modified config
TEMP_CONFIG="temp_config.json"

# Ask for new values for the config file
read -p "Insert the dataset name to use for the downstream task ('chest' or 'hand' or 'cephalo'): " DATASET_NAME

# Assert that the dataset name is valid
if [ "$DATASET_NAME" != "chest" ] && [ "$DATASET_NAME" != "hand" ] && [ "$DATASET_NAME" != "cephalo" ]; then
    echo "Invalid dataset name. Please choose 'chest' or 'hand' or 'cephalo'."
    exit 1
fi

# Ask for the model to be used for the pre-training task
read -p "Insert the name of the model to be used for the pre-training task ('ddpm' or 'moco' or 'mocov2' or 'mocov3' or 'simclr' or 'simclrv2' or 'dino' or 'barlow_twins' or 'byol'): " MODEL_NAME

# Assert that the model name is valid 
if [ "$MODEL_NAME" == "ddpm" ]; then
    # Load the original config
    jq --arg DATASET_NAME "$DATASET_NAME" '.dataset.name = $DATASET_NAME' $DDPM_CONFIG_PATH > $TEMP_CONFIG

    # Execute the main.py script with the modified config
    python $DDPM_PY_PATH --config $TEMP_CONFIG

elif [ "$MODEL_NAME" == "moco" ] || [ "$MODEL_NAME" == "mocov2" ] || [ "$MODEL_NAME" == "mocov3" ] || [ "$MODEL_NAME" == "simclr" ] || [ "$MODEL_NAME" == "simclrv2" ] || [ "$MODEL_NAME" == "dino" ] || [ "$MODEL_NAME" == "barlow_twins" ] || [ "$MODEL_NAME" == "byol" ]; then

    # Ask for the backbone encoder to be used for the pre-training task
    read -p "Insert the name of the backbone encoder to be used for the pre-training task ('resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'): " BACKBONE_ENCODER

    # Assert that the backbone encoder is valid
    if [ "$BACKBONE_ENCODER" != "resnet18" ] && [ "$BACKBONE_ENCODER" != "resnet50" ] && [ "$BACKBONE_ENCODER" != "resnet101" ] && [ "$BACKBONE_ENCODER" != "resnet152" ] && [ "$BACKBONE_ENCODER" != "resnext50_32x4d" ] && [ "$BACKBONE_ENCODER" != "resnext101_32x8d" ] && [ "$BACKBONE_ENCODER" != "vgg11" ] && [ "$BACKBONE_ENCODER" != "vgg13" ] && [ "$BACKBONE_ENCODER" != "vgg16" ] && [ "$BACKBONE_ENCODER" != "vgg19" ] && [ "$BACKBONE_ENCODER" != "densenet121" ] && [ "$BACKBONE_ENCODER" != "densenet169" ] && [ "$BACKBONE_ENCODER" != "densenet201" ] && [ "$BACKBONE_ENCODER" != "densenet161" ]; then
        echo "Invalid backbone encoder. Please choose 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'."
        exit 1
    fi

    # Load the original config
    jq --arg BACKBONE_ENCODER "$BACKBONE_ENCODER" --arg MODEL_NAME "$MODEL_NAME" --arg DATASET_NAME "$DATASET_NAME" '.dataset.name = $DATASET_NAME | .model.encoder = $BACKBONE_ENCODER | .model.name = $MODEL_NAME' $SSL_CONFIG_PATH > $TEMP_CONFIG

    # Execute the main.py script with the modified config
    python $SSL_PY_PATH --config $TEMP_CONFIG

else
    echo "Invalid model name. Please choose 'ddpm' or 'moco' or 'mocov2' or 'mocov3' or 'simclr' or 'simclrv2' or 'dino' or 'barlow_twins' or 'byol'."
    exit 1
fi

# Remove the temporary config file
rm $TEMP_CONFIG