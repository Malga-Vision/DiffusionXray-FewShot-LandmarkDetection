#!/bin/bash

# Path to the config.json file
CONFIG_PATH="downstream_task/config/config.json"

# Path to the main.py script
MAIN_PY_PATH="downstream_task/main.py"

# Temporary file for modified config
TEMP_CONFIG="temp_config.json"

# Ask for new values for the config file 
read -p "Insert the dataset name to use for the downstream task ('chest' or 'hand' or 'cephalo'): " DATASET_NAME

# Assert that the dataset name is valid
if [ "$DATASET_NAME" != "chest" ] && [ "$DATASET_NAME" != "hand" ] && [ "$DATASET_NAME" != "cephalo" ]; then
    echo "Invalid dataset name. Please choose 'chest' or 'hand' or 'cephalo'."
    exit 1
fi

read -p "Insert the name of the model to be used for the downstream task ('ddpm' or 'imagenet' or 'moco' or 'mocov2' or 'mocov3' or 'simclr' or 'simclrv2' or 'dino' or 'barlow_twins' or 'byol'): " MODEL_NAME

# If model_name is "ddpm", then ask for the pretrained model path and if the user is tuning the DDPM pre-training iterations
if [ "$MODEL_NAME" == "ddpm" ]; then
    BACKBONE_ENCODER=""

    read -p "Insert the path to the pretrained ddpm model for the downstream task: " PRETRAINED_MODEL_PATH

    # Ask if the user is tuning the DDPM pre-training iterations
    read -p "Are you tuning the DDPM pre-training iterations? (true or false): " USE_VAL_SET

    # Assert that the value is valid
    if [ "$USE_VAL_SET" != "true" ] && [ "$USE_VAL_SET" != "false" ]; then
        echo "Invalid value for tuning DDPM. Please choose 'true' or 'false'."
        exit 1
    fi 

    # Ask if the dataset used for the downstream task is different from the one used for pretraining
    read -p "Is the dataset used for the downstream task different from the one used for pretraining? (true or false): " DIFFERENT_DATASET

    # Assert that the value for different dataset is valid
    if [ "$DIFFERENT_DATASET" == "true" ]; then
        DIFFERENT_DATASET=true
    elif [ "$DIFFERENT_DATASET" == "false" ]; then
        DIFFERENT_DATASET=false
    else
        echo "Invalid value for different dataset. Please choose 'true' or 'false'."
        exit 1
    fi

elif [ "$MODEL_NAME" == "imagenet" ]; then
    PRETRAINED_MODEL_PATH=""
    USE_VAL_SET=false
    DIFFERENT_DATASET=false

    # Ask for the backbone encoder to be used for the downstream task
    read -p "Insert the name of the backbone encoder to be used for the downstream task ('resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'): " BACKBONE_ENCODER

    # Assert that the backbone encoder is valid
    if [ "$BACKBONE_ENCODER" != "resnet18" ] && [ "$BACKBONE_ENCODER" != "resnet50" ] && [ "$BACKBONE_ENCODER" != "resnet101" ] && [ "$BACKBONE_ENCODER" != "resnet152" ] && [ "$BACKBONE_ENCODER" != "resnext50_32x4d" ] && [ "$BACKBONE_ENCODER" != "resnext101_32x8d" ] && [ "$BACKBONE_ENCODER" != "vgg11" ] && [ "$BACKBONE_ENCODER" != "vgg13" ] && [ "$BACKBONE_ENCODER" != "vgg16" ] && [ "$BACKBONE_ENCODER" != "vgg19" ] && [ "$BACKBONE_ENCODER" != "densenet121" ] && [ "$BACKBONE_ENCODER" != "densenet169" ] && [ "$BACKBONE_ENCODER" != "densenet201" ] && [ "$BACKBONE_ENCODER" != "densenet161" ]; then
        echo "Invalid backbone encoder. Please choose 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'."
        exit 1
    fi


elif [ "$MODEL_NAME" == "moco" ] || [ "$MODEL_NAME" == "mocov2" ] || [ "$MODEL_NAME" == "mocov3" ] || [ "$MODEL_NAME" == "simclr" ] || [ "$MODEL_NAME" == "simclrv2" ] || [ "$MODEL_NAME" == "dino" ] || [ "$MODEL_NAME" == "barlow_twins" ] || [ "$MODEL_NAME" == "byol" ]; then
    # Ask for the backbone encoder to be used for the downstream task
    read -p "Insert the name of the backbone encoder to be used for the downstream task ('resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'): " BACKBONE_ENCODER

    # Assert that the backbone encoder is valid
    if [ "$BACKBONE_ENCODER" != "resnet18" ] && [ "$BACKBONE_ENCODER" != "resnet50" ] && [ "$BACKBONE_ENCODER" != "resnet101" ] && [ "$BACKBONE_ENCODER" != "resnet152" ] && [ "$BACKBONE_ENCODER" != "resnext50_32x4d" ] && [ "$BACKBONE_ENCODER" != "resnext101_32x8d" ] && [ "$BACKBONE_ENCODER" != "vgg11" ] && [ "$BACKBONE_ENCODER" != "vgg13" ] && [ "$BACKBONE_ENCODER" != "vgg16" ] && [ "$BACKBONE_ENCODER" != "vgg19" ] && [ "$BACKBONE_ENCODER" != "densenet121" ] && [ "$BACKBONE_ENCODER" != "densenet169" ] && [ "$BACKBONE_ENCODER" != "densenet201" ] && [ "$BACKBONE_ENCODER" != "densenet161" ]; then
        echo "Invalid backbone encoder. Please choose 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'densenet161'."
        exit 1
    fi

    read -p "Insert the path to the $BACKBONE_ENCODER backbone model pre-trained with $MODEL_NAME: " PRETRAINED_MODEL_PATH
    USE_VAL_SET=false
    DIFFERENT_DATASET=false

else
    echo "Invalid model name. Please choose 'ddpm' or 'moco' or 'mocov2' or 'mocov3' or 'simclr' or 'simclrv2' or 'dino' or 'barlow_twins' or 'byol'."
    exit 1
fi

# Assert that the path to the pretrained model is valid
if [ "$PRETRAINED_MODEL_PATH" != "" ] && [ ! -f $PRETRAINED_MODEL_PATH ]; then
    echo "Invalid path to the pretrained model."
    exit 1
fi


read -p "Insert the number of training labeled samples to be used for the downstream task ('all' or a number): " TRAINING_SAMPLES

# Assert that the number of training samples is valid
if [ "$TRAINING_SAMPLES" != "all" ] && ! [[ "$TRAINING_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Invalid number of training samples. Please choose 'all' or a number."
    exit 1
fi

# Load the original config
#jq '.model.name = $MODEL_NAME | .training_protocol.finetuning.path = $PRETRAINED_MODEL_PATH | .training_protocol.finetuning.different_dataset = $DIFFERENT_DATASET | .inference_protocol.use_validation_set_for_inference = $USE_VAL_SET | .dataset.name = $DATASET_NAME | .dataset.training_samples = $TRAINING_SAMPLES' $CONFIG_PATH > $TEMP_CONFIG
jq --arg BACKBONE_ENCODER "$BACKBONE_ENCODER" --arg MODEL_NAME "$MODEL_NAME" --arg PRETRAINED_MODEL_PATH "$PRETRAINED_MODEL_PATH" --arg DIFFERENT_DATASET "$DIFFERENT_DATASET" --arg USE_VAL_SET "$USE_VAL_SET" --arg DATASET_NAME "$DATASET_NAME" --arg TRAINING_SAMPLES "$TRAINING_SAMPLES" '.model.name = $MODEL_NAME | .model.encoder = $BACKBONE_ENCODER | .training_protocol.finetuning.path = $PRETRAINED_MODEL_PATH | .training_protocol.finetuning.different_dataset = $DIFFERENT_DATASET | .inference_protocol.use_validation_set_for_inference = $USE_VAL_SET | .dataset.name = $DATASET_NAME | .dataset.training_samples = $TRAINING_SAMPLES' $CONFIG_PATH > $TEMP_CONFIG

# Execute the main.py script with the modified config
python $MAIN_PY_PATH --config $TEMP_CONFIG

# Remove the temporary config file
rm $TEMP_CONFIG