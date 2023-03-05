# TLRel
Source code of the paper "A Novel Tensor Learning Model for Joint Relational Triplet Extraction"



#### Requirement:

```
  python==3.8.8
  torch==1.6.0
```

#### Dataset and Pre-trained Embedding:
Pre-trained Glove 840B Embedding: Download from https://nlp.stanford.edu/projects/glove/ 

Dataset Download from https://github.com/NKU-IIPLab/SMHSA.


#### How to run:
```
  python dataProcess.py # For preprocessing dataset
  python loadPretrainEmbedding.py # For loading pre-trained embedding 
  python main.py --mode train # For training
  python main.py --mode test --test_model ./test_model/modelFinal.model # For testing
```
