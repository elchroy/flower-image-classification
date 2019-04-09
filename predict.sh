#!/bin/sh
python predict.py "./checkpoints_dir/__checkpoint_alexnet.pth" --path_to_dir "flowers/test/1" --top_k 1 --category_names cat_to_name.json

