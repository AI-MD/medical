#!/bin/env bash

python multi_class.py /root/dataset/Osteoporosis/Sagittal_Contrast_Final_1/ --n_gpu 4 -b 32 --lr 0.0001 -j 16 --epochs 60 --seed 0 --log logs/CE
python multi_class.py /root/dataset/Osteoporosis/Sagittal_Contrast_Final_1/ --n_gpu 4 -b 32 --lr 0.0001 -j 16 --epochs 60 --seed 1 --log logs/CE
python multi_class.py /root/dataset/Osteoporosis/Sagittal_Contrast_Final_1/ --n_gpu 4 -b 32 --lr 0.0001 -j 16 --epochs 60 --seed 2 --log logs/CE