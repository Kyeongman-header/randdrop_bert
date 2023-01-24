from random import sample
import tensorflow as tf

import os
import numpy as np
# from transformers import pipeline
# from summarizer import Summarizer
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange
import random
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)

RANGE=10
START=4001
T="train"


total_source=[]
with open("writingPrompts/"+ T +".wp_source", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_source.append(temp_stories)

total_target=[]

with open("writingPrompts/"+ T +".wp_target", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_target.append(temp_stories)

summary_prefix_target=[]
summary=[]
truncated_target=[]
#truncated_target=[]
whole_data=[]
#print(START)
#print(RANGE)
#print(len(total_target[0]))
#print(len(total_target[0][START:RANGE]))
if RANGE != 0:
    whole_data=total_target[0][START:START+RANGE]
else:
    whole_data=total_target[0]
max_sum=0
max_target=0
print("whole data: " + str(len(whole_data)))
seq_arr=[]
lens=[]


neg_examples=[]
pos_examples=[]
for t in tqdm(whole_data):
    # t에서 ;이랑 .. 같은 애들 . 으로 바꿔야 겠다
    

    t=t.replace(".....",".")
    t=t.replace("....",".")
    t=t.replace("...",".")
    t=t.replace("..",".")
    t=t.replace(";",";.")
    t=t.replace("!","!.")
    t=t.replace("?","?.")
    t=t.replace("\""," ")
    t=t.replace("\""," ") 
    t=t.replace('“'," ")
    t=t.replace('”'," ")
    t=t.replace('"'," ")
    t=t.replace("*","")
    t=t.replace(". ''","'' .")

    
    token_len=len(tokenizer(t,return_tensors="tf").input_ids[0])
    if token_len>=1000:
        print("This text is too long >1020.")
        continue
    else:
        print("Token length is : " + str(token_len))
    
    tt=t.split('.')

    # print(tt)
    # print(len(tt))

    # rand_rate=random.uniform(0.8, 1)
    # drop_1=sample(tt, round(len(tt)*rand_rate))

    d_arr=sample(range(0, len(tt)), round(len(tt)*0.2))
    # print(d_arr)
    new_tt=[]
    j=0
    d_count=0
    for s in tt:

        if len(d_arr)>d_count and d_arr[d_count] == j:
            j=j+1
            d_count+=1
            continue
        else:
            new_tt.append(s)
            j=j+1
    drop_1=new_tt

    # print(drop_1)
    # print(len(drop_1))

    d_arr=sample(range(0, len(tt)), round(len(tt)*0.1))
    # print(d_arr)
    new_tt=[]
    j=0
    d_count=0
    for s in tt:

        if len(d_arr)>d_count and d_arr[d_count]-2 < j and j < d_arr[d_count]+2:
            j=j+1
            continue
        elif len(d_arr)>d_count and d_arr[d_count]+2<=j:
            d_count+=1
            j=j+1
            continue
        else:
            new_tt.append(s)
            j=j+1
    drop_2=new_tt

    # print(drop_2)
    # print(len(drop_2))

    rand_rate=random.uniform(0, 0.2)
    drop_3=tt[:-round(len(tt)*rand_rate)]

    # print(drop_3)
    # print(len(drop_3))

    rand_rate=random.uniform(0, 0.2)
    drop_4=tt[round(len(tt)*rand_rate):]

    # print(drop_4)
    # print(len(drop_4))

#     # 4가지 버전의 neg examples.
#     # 각각 완전 random한 drop과, random drop이 앞뒤 2문장씩 뭉탱이로 일어나는 drop과, 뒤 쪽 부분이 잘려나간 drop과, 앞쪽 부분이 잘려나간 drop이다.
    dropped=random.choice([drop_1,drop_2,drop_3,drop_4])
    
    print(len(dropped))
    print(len(tt))

    neg_ex=('.').join(dropped)
    pos_ex=('.').join(tt)
    neg_ex=neg_ex.replace(".","[SEP]")
    pos_ex=pos_ex.replace(".","[SEP]")

    # neg_ex.replace(';.',";")
    # neg_ex.replace("?.","?")
    # neg_ex.replace("!.","!")
    # pos_ex.replace(';.',";")
    # pos_ex.replace("?.","?")
    # pos_ex.replace("!.","!")
    
    print("neg : " + neg_ex)
    print("pos : " + pos_ex)
    

    neg_examples.append(neg_ex) # 넷 중 하나를 고른다.'
    pos_examples.append(pos_ex)

print(len(neg_examples))
print(len(pos_examples))

token_neg=tokenizer(neg_examples,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids
token_pos=tokenizer(pos_examples,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids

npneg=np.array(neg_examples)
nppos=np.array(pos_examples)
nptoken_neg=token_neg.numpy()
nptoken_pos=token_pos.numpy()

file="contrastive_dataset"

createFolder("npdata/"+file)

print("is this npdata exists already? => ")
print(os.path.isfile("./npdata/"+file+"/npneg.npy"))

if os.path.isfile("./npdata/"+file+"/npneg.npy"):
    past=np.load("./npdata/"+file+"/npneg.npy")
    npneg=np.concatenate((past,npneg),axis=0)
np.save("./npdata/"+file +"/npneg",npneg)

if os.path.isfile("./npdata/"+file+"/nppos.npy"):
    past=np.load("./npdata/"+file+"/nppos.npy")
    nppos=np.concatenate((past,nppos),axis=0)
np.save("./npdata/"+file +"/nppos",nppos)

if os.path.isfile("./npdata/"+file+"/nptoken_neg.npy"):
    past=np.load("./npdata/"+file+"/nptoken_neg.npy")
    nptoken_neg=np.concatenate((past,nptoken_neg),axis=0)
np.save("./npdata/"+file +"/nptoken_neg",nptoken_neg)

if os.path.isfile("./npdata/"+file+"/nptoken_pos.npy"):
    past=np.load("./npdata/"+file+"/nptoken_pos.npy")
    nptoken_pos=np.concatenate((past,nptoken_pos),axis=0)
np.save("./npdata/"+file +"/nptoken_pos",nptoken_pos)




