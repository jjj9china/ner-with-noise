# ner-with-noise
- This is an NER tools which can help you train better chinese ner model when facing noisy data.
- 本项目旨在提供一个中文带噪NER的训练工具箱。

NER任务作为NLP领域的一个基础任务，在神经网络大肆盛行的今天，似乎快要被人们遗忘了。
不可否认的是，BiLSTM-CRF已成为这类任务的标配，一般情况下，使用该模型能解决90%的问题。
但是想要轻松的应对这个领域的其他问题：如何解决数据存在噪声、数据量过少、实体嵌套、非连续实体、联合关系抽取等问题，似乎还未有定论。

本项目旨在帮助研究者或者开发者在面对**数据质量**问题时，提供一个简单易用的工具箱。

本项目实现以下几种相关方法：
|NAME|PAPER|STATUS|
|:--|:--|:--|
|BiLSTM/BiLSTM-CRF|(Baseline Model) (2016 NAACL) [Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)|Done|
|Partical-CRF/Fuzzy-CRF|(2007 AAAI) [Learning extractors from unlabeled text using relevant databases](https://www.aaai.org/Papers/Workshops/2007/WS-07-14/WS07-14-002.pdf)|Done|
|MentorNet|(2018 ICML) [MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels](https://arxiv.org/abs/1712.05055)|Done|
|Positive-Unlabeled Learning|(2019 ACL) [Distantly Supervised Named Entity Recognition using Positive Unlabeled Learning](https://arxiv.org/pdf/1906.01378.pdf)|Coming Soon|
|CrossWeigh|(2019 EMNLP) [CrossWeigh Training Named Entity Tagger from Imperfect Annotations](https://arxiv.org/pdf/1909.01441)|Done|
|Marginal Likelihood CRF|(2018 EMNLP) [Marginal Likelihood Training of BiLSTM-CRF for Biomedical Named Entity Recognition from Disjoint Label Sets](https://www.aclweb.org/anthology/D18-1306.pdf)|Coming Soon|

