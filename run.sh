#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2
echo $CUDA_VISIBLE_DEVICES

torchrun --standalone --nproc_per_node 2 -m src.main --task handwriting-txt2img_with_style --neptune_project_name handwriting --data_type full --run_id test --model_id 12
