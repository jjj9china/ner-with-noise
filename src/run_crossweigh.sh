#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

python ./crossweigh/split.py --input_files ../data/train.col \
                             --output_folder ../data/ \
                             --splits 5 \
                             --folds 5 \
                             --schema sio

export DATA_FOLDER=../data
export LOG_FOLDER=../log
export SAVED_MODEL_PATH=../saved_model
# training each split/fold
for splits in $(seq 0 1 4); do
    for folds in $(seq 0 1 4); do
        FOLD_FOLDER=split-${splits}/fold-${folds}
        python train_baseline.py --train_file ${DATA_FOLDER}/${FOLD_FOLDER}/train.col \
                                 --dev_file ${DATA_FOLDER}/${FOLD_FOLDER}/dev.col \
                                 --test_file ${DATA_FOLDER}/test.col \
                                 --gaz_file ${DATA_FOLDER}/wv_txt.txt \
                                 --model_save_path ${SAVED_MODEL_PATH}/ \
                                 --model_name s-${splits}-f-${folds} \
                                 --output_path ${DATA_FOLDER}/${FOLD_FOLDER}/ \
                                 --log_path ${LOG_FOLDER}/ \
                                 --use_crf True
        sleep 1h
    done
done

# collect and generate data
python ./crossweigh/collect.py --split_folders ../data/split-\* \
                               --split_info info.json\
                               --splits 5 \
                               --folds 5 \
                               --origin_files ${DATA_FOLDER}/train.col \
                               --origin_file_schema sio \
                               --output_folder ${DATA_FOLDER}/ \
                               --output_file unknown_tag.col \
                               --output_json output_json.json \
                               --model_predicted_filename predict.col

