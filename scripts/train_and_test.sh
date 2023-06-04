#!/usr/bin/env bash

cd $(dirname $0)/../

DATASET_NAME=$1

./train_ace.py datasets/${DATASET_NAME} output/${DATASET_NAME}/map_parameters.pt
./test_ace.py datasets/${DATASET_NAME} output/${DATASET_NAME}/map_parameters.pt --render_visualization=True
