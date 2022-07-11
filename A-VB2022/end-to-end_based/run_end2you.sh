#!/bin/bash

if [[ $# -ne 2 ]];then
    echo "Please provide the path to End2You files (1st arg), and the type of task (2nd arg) you are running."
    exit
fi

if [ "$2" == "type" ]; then
    loss=ce
    metric=uar
    num_outputs=8
else
    loss=mse
    metric=ccc
    if [ "$2" == "high" ]; then
        num_outputs=10
    elif [ "$2" == "low" ]; then
        num_outputs=2
    elif [ "$2" == "culture" ]; then
        num_outputs=40
    fi
fi


save_path=$1

partitions=("train" "val" "test")

wget https://raw.githubusercontent.com/end2you/end2you/master/docs/cli/main.py

partitions=("train" "val" "test")
# Start generating hdf5 data for all partitions
for p in ${partitions[@]}; do
    python main.py --modality="audio" \
                   --root_dir=$save_path/data \
                   generate \
                   --input_file=$save_path/labels/"$p"_input_file.csv \
                   --save_data_folder=$save_path/data/$p
done

# Start training
python main.py --modality="audio" \
               --root_dir=./training \
               --batch_size=8 \
               --model_name=emo18 \
               --num_outputs=$num_outputs \
               --take_last_frame="true" \
               train  \
               --loss=$loss \
               --metric=$metric \
               --train_dataset_path=$save_path/data/train \
               --valid_dataset_path=$save_path/data/val \
               --num_epochs=30 \
               --learning_rate=0.0001

# Start evaluation
python main.py --modality="audio" \
               --root_dir=./ \
               --model_name=emo18 \
               --num_outputs=$num_outputs \
               --take_last_frame="true" \
               test  \
               --prediction_file=predictions.csv \
               --metric=$metric \
               --dataset_path=$save_path/data/test \
               --model_path=./training/model/best.pth.tar
