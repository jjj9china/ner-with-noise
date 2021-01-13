各模块的说明：

- **baseline**: basic NER models: bilstm and bilstm-crf.
- **coteach**: Coteach model.
- **crossweigh**: CrossWeigh model.
- **mentornet**: MentorNet model.
- **partialcrf**: Partial/Fuzzy-CRF model.
- **preprocess**: data preprocessing tools.
- **utils**: tools in model train/test.

其中，除了`utils`和`preprocess`属于公共模块，其余各模型为相应论文的具体实现。
不同模块的训练代码已经封装在`train_*.py`中，准备相应的数据后，可直接训练。
