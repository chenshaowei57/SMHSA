# SMHSA
Source code of the paper "Attention as Relation: Learning Supervised Multi-head Self-Attention for Relation Extraction, IJCAI 2020."



#### Requirement:

```
  python==3.6.9
  torch==0.4.0
```

#### Dataset and Pre-trained Embedding:
Pre-trained Glove 840B Embedding: Download from https://nlp.stanford.edu/projects/glove/ 

Dataset Download from https://drive.google.com/drive/folders/1VJVC18lPibxXCQOKN0BAKxgGm2qUXZHh?usp=sharing.


#### How to run:
```
  python dataProcess.py # For preprocessing dataset
  python loadPretrainEmbedding.py # For loading pre-trained embedding 
  python main.py --mode train # For training
  python main.py --mode test --test_model ./test_model/modelFinal.model # For testing
```
