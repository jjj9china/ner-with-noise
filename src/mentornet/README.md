### 概述
实现了论文[MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels](https://arxiv.org/abs/1712.05055) 中的带噪训练方法。

`student.py` 封装了多种学生网络模型；
`mentor.py` 实现了多种导师网络模型；

### 使用说明
具体使用方法已封装在`train_mentornet.py`中。

在细节上与原始MentorNet论文稍有不同：
|Diff|原论文|本实现|
|:--|:--|:--|
|样本粒度|句子|词|
|mentor训练阶段|student训练的特定的阶段|student与mentor交替训练|
|mentor使用特征|student的loss、loss-diff、当前epoch等|student的隐层输出|
|mentor预测如何使用|mentor的输出直接与student的loss进行加权|根据mentor的输出，**修改label**，构建unknown_tag|

上述几种改进，已在作者私有数据中得到验证，读者可根据实际问题亲自试验效果。

### NOTE
1. 训练`mentor`时，数据及其不均衡（正常`token`的数量远多于噪声`token`），注意选择合适的`pos_weight`;
2. 训练`mentor`时，若发现`mentor`效果不是很理想，注意适当调整`config.py`中的`predict_threshold`，提升`mentor`在**噪声类**上的准确率；
3. 提升`mentor`在噪声类上的准确率，意味着`mentor`预测一个词语为噪声`token`时错误较少，对原始数据的修改比例也较少，更容易提升`student`的效果；
一种极端情况是，调整`predict_threshold`阈值，使得`mentor`输出结果全部为正常token，不对输入标签做任何修改，此时`MentorNet`的作用退化为只训练`student`。

