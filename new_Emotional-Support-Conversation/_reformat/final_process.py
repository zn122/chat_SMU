#아연수정
import json
import random
import csv
import copy
import numpy as np
import pandas as pd

# 대화 중간 잘라서 앞부분 init
# 뒷부분 final 


with open('ESConv.json', 'r', encoding="UTF-8") as f:
    jcorp = json.load(f)

with open('labelling_sentiment_chatbot_dataset.csv', newline='', encoding="UTF-8") as f:
    reader = csv.reader(f)
    tag_data = list(reader)

tag = copy.deepcopy(tag_data[1:])  # remove 'sentiment_chatbot_dataset' column name

# print(tag) # sentiment_chatbot_dataset

# start and end index of dialog
tag_idx = list()
for i, t in enumerate(tag):
    if t[0].split()[2] == '0': # 첫번째 발화점이라면, 대화가 시작되는 부분.
        tag_idx.append(i) #대화가 시작되는 부분의 진짜 index를 저장해둔다.

s_e = list()
for j in range(1300):
    if j != 1299:
        s_e.append((tag_idx[j], tag_idx[j+1]-1))
    else:
        s_e.append((tag_idx[j], len(tag)-1))
        
# s_e 는 (대화가 시작된 지점의 index, 대화가 끝난 지점의 index)

# emotion tag
emt = {'anger': 0, 'anxiety': 0, 'depression': 0, 'disgust': 0, 'fear': 0, 'guilt': 0,
       'jealousy': 0, 'nervousness': 0, 'pain': 0, 'sadness': 0, 'shame': 0, 'neutral': 0}
intensity ={'111':0,'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,}

ds = list()

for cnt in range(len(jcorp)): #
    d = dict()
    u_emo = 'neutral'
    emt_ = copy.deepcopy(emt)
    intensity_ = copy.deepcopy(intensity)
    d["emotion_type"] = jcorp[cnt]["emotion_type"]
    d["problem_type"] = jcorp[cnt]["problem_type"]
    d["situation"] = jcorp[cnt]["situation"]

    

    d_list = list()
    s, e = s_e[cnt][0], s_e[cnt][1] #s와 e는 index를 의미

    for n in range(e - s + 1):
        tmp = dict()
        if tag[s][2] != '': # tag[s][2] = text : "content": "Hello, what would you like to talk about?"
            if tag[s][1] == '': # usr 
                tmp["text"] = tag[s][2]
                tmp["speaker"] = "usr"
                emt_[tag[s][3]] += 1 # 감정이 몇 개 있는지 count, tag[s][3] 
                intensity_[str(int(float(tag[s][6])))] += 1 # intensity 몇 개 있는지 count, tag[s][3] 
                
                if (s < e and tag[s + 1][1] != '') or s == e:
                    u_emo = max(emt_, key=emt_.get) #두줄 있는 경우
                    u_intensity = max(intensity_, key=intensity_.get) #두줄 있는 경우
                    u_intensity = int(float(tag[s][6]))
                    if u_emo == 'neutral': 
                        emt_.pop('neutral')
                        if emt_[max(emt_, key=emt_.get)] != 0:
                            u_emo = max(emt_, key=emt_.get)
                    intensity_ = copy.deepcopy(intensity)
                    emt_ = copy.deepcopy(emt)
                
                
            else: #sys
                tmp["text"] = tag[s][2]
                tmp["speaker"] = "sys"
                tmp["strategy"] = tag[s][1]
                tmp["emotion"] = u_emo
                tmp["intensity"] = str(u_intensity)
                print(tmp["intensity"])
        
            d_list.append(tmp)
        s += 1


    d["dialog"] = d_list
    

    ds.append(d)
    #print(ds)

# df = pd.DataFrame({ 'emotion':emotion_count,
#                     'initial_intensity':initial_intensity_count,
#                     'final_intensity':final_intensity_count,
#                     'change_intensity':change_intensity})
# df.to_csv('process_result.csv', index=False)

# shuffle dataset
random_seed = 42
random.seed(random_seed)
random.shuffle(ds)

# split and write files
# train : vaild : test = 0.7 : 0.15 : 0.15
f = open('final_set/train.txt', 'w+', encoding='utf-8')
for d_dict in ds[:910]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()

f = open('final_set/valid.txt', 'w+', encoding='utf-8')
for d_dict in ds[910:1105]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()

f = open('final_set/test.txt', 'w+', encoding='utf-8')
for d_dict in ds[1105:]:
    tmp = json.dumps(d_dict)
    f.write(tmp+'\n')
f.close()


# json 파일로 test 저장
with open('final_set/test.json', 'w') as f : 
	json.dump(ds[1105:], f, indent=4)


