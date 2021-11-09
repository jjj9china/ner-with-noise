本模块主要基于论文 [Learning extractors from unlabeled text using relevant databases](https://www.aaai.org/Papers/Workshops/2007/WS-07-14/WS07-14-002.pdf)，
实现了bilstm-partial-crf模型。

模型在计算loss时，ground truth路径为所有可能的路径，而不仅仅是一条路径。我们知道CRF的路径得分计算方法是使用动态规划实现的，沿着时间维度依此计算。某时刻在面对多标签时，计算多条目标路径得分；面对未知标签时，计算所有目标路径得分。

由于数据的读入是在`preprocess`模块中实现的，因此，使用该方法时，需要修改`preprocess/config.py`中相关配置:

```
UNKNOWN_LABEL = 'unknown_label'
MULTI_LABEL_SPLIT_TAG = '|'
```
