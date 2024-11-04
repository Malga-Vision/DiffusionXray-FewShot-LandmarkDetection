#!/bin/bash
# Check if more than one gpu is available, if so ask for which one to use
if [ $(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l) -gt 1 ]; then
    read -p "Insert the GPU number to use: " GPU
    export CUDA_VISIBLE_DEVICES=$GPU
fi

# Ask for which experiment to run
read -p "Insert the experiment to run (1 for pre-training task, 2 for ImageNet comparative study, 3 for downstream task experiments): " EXPERIMENT

# Execute the corresponding experiment
case $EXPERIMENT in 
    1)
        bash experiments/launch_pretraining.bash
        ;;
    2)
        bash experiments/launch_imagenet_comparative_study.bash
        ;;
    3)
        bash experiments/launch_landmarks_experiments.bash
        ;;
    *)

        echo "Invalid experiment number" 
        ;;
esac



