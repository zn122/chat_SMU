# Emotional-Support-Conversation
https://github.com/ChaeheePark/XAI-Emotionally-Supportive-Conversations

https://github.com/thu-coai/Emotional-Support-Conversation

### *Copyright © 2021 CoAI Group, Tsinghua University. All rights reserved. Data and codes are for academic research use only.*

Data and codes for the ACL 2021 paper: [**Towards Emotional Support Dialog Systems**](https://arxiv.org/abs/2106.01144)
If you use our codes or your research is related to our paper, please kindly cite our paper:

```bib
@inproceedings{liu-etal-2021-towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang  and 
    Zheng, Chujie  and 
    Demasi, Orianna  and 
    Sabour, Sahand  and 
    Li, Yu  and 
    Yu, Zhou  and 
    Jiang, Yong  and 
    Huang, Minlie},
  booktitle={Proceedings of the 59th annual meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Data

The corpus file is `ESConv.json`. We have collected **more** conversations with more problem topics. ESConv now contians 1,300 conversations with 10 topic problems.

`sentiment_chatbot_dataset_csv` : Add an emotion tag for each text.

### Statistics
#### Problem Category

| Problem Category | ongoing depression | breakup with partner | job crisis | problems with friends | academic pressure | procras-<br>tination* | alcohol abuse* | issues with parent* | sleep problems* |  appearance anxiety* | school bullying* | issues with children* |
| :-------- | :---------- | :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- | :---------- | :---------- | 
| Number| 351 | 239 | 280 | 179 | 156 |  13 | 12 | 18 | 28 | 12 | 2 | 10 |

#### Emotion Category
anger, anxiety, depression, disgust, fear, guilt, jealousy, nervousness, pain, sadness, shame, neutral

\* denotes the new topics added during the second collection. We hope new data supports the future research in transferring the ability of models from old topics to new ones. 

<font size=1>

#### Strategy Category 
| Strategy Category| Number   |
| :--------------  | :------- |
| Questions | 3801(20.7%)|
| Self-disclosure | 1713(9.3%) |
| Affirmation and Reassurance | 2827(15.4%) |
| Providing Suggestions | 2954(16.1%) |
| Other | 3661(18.3%) / 3341(18.2%) |
| Reflection of feelings |  1436(7.8%) | 
| Information | 1215(6.6%) | 
| Restatement or Paraphrasing | 1089(5.9%) |

</font>

## Preparing Enviroment

```bash
conda env create -f env.yml -n cuda
conda activate cuda
```
or 

#### In CoLAB

`chatbot_train & run.ipynb` You must run the cells in the this ipynb file to set Python == 3.7, torch == 1.7.1, and transformers == 4.9.2.

## Model
You can check the model used in the `Blenderbot_small-90M` folder.


[previous]

[BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M)
If you would like to evaluate generated results with Embedding-based similarity, you can download my prepared embedding files from [here](https://1drv.ms/f/s!Aky8v8NZbQx1qj7OlJKcQEJ6qrWm).

## Preprocessing &  Training 
`!python prepare.py`, `!python train.py in file` in `chatbot_train & run.ipynb` 

## Inference with Your Model
every 10,20,30,40,50 epochs of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. 

