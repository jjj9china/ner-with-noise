实现了 CrossWeigh: https://arxiv.org/abs/1909.01441 的数据分割（split.py）和模型预测结果统计（collect.py）。
未提供模型训练的代码，用户可根据需要自行选择。推荐的做法是使用**baseline**中的bilstm 或者 bilstm-crf 模型。

split.py 将输入数据文件进行n次k-fold分割，每次k-fold的文件保存在`split-*`文件夹里；
         每个fold的训练集和验证集的相关信息保存在info.json里。
collect.py 统计n次k-fold模型训练的预测结果；并提供多种策略：样本过滤、标签修改（unknown-tag or multi-tag）

Example:

- step1
```
python split.py --input_files YOUR_DATA_PATH \
                --output_folder YOUR_OUTPUT_PATH \
                --shema YOUR_DATA_LABEL_SCHEMA
```

- step2
```
Train your model and save predict result on dev data.
```

- step3
```
python collect.py --split_folders YOUR_OUTPUT_PATH \
                  --origin_files YOUR_DATA_PATH \
                  --origin_file_schema YOUR_DATA_LABEL_SCHEMA \
                  --output_folder YOUR_OUTPUT_PATH \
                  --output_file YOUR_OUTPUT_FILE_NAME \
                  --model_predicted_filename YOUR_MODEL_PREDICT_FILE_NAME
```
