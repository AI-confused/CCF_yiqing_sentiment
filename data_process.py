import pandas

#读取原始训练集，修改title名字
train_m = pd.read_csv('train_labled.csv')
train_m.columns = ['id','time','account','text','pic_url','video_url','lable']
#去除不需要的字段内容，保留id,text,lable
train_m = train_m.drop(['time','account','pic_url','video_url'],axis=1)
#筛选text & lable字段非空 & text长度超过10的数据
train_m = train_m[train_m['text'].notna()&train_m['lable'].notna()&train_m['text'].str.len()>10]
#分析数据，计算最大、最小、平均文本长度
min_t, max_t = 100,0
ave = 0
num = 0
text = train_m.iloc[:,3].values
for i,t in enumerate(text):
    length = len(t)
    min_t = min(min_t, length)
    max_t = max(max_t, length)
    ave += length/(1e5-354)
    
print(ave, min_t, max_t,num)
#去除lable不是-1，0，1的数据
for i in train_m.index:
    if train_m.iloc[i]['lable'] not in ['0','-1','1']:
        x.append(i)
train_m.drop(x, inplace=True)
#把lable字段修改为int类型，并且将-1,0,1修改为0,1,2。适应bert
new_lables = train_m.iloc[:,-1].values
df = train_m.drop(['lable'],axis=1)
print(df)
for i,l in enumerate(new_lables):
    if l=='-1':
        new_lables[i] = 0
    elif l=='0':
        new_lables[i] = 1
    else:
        new_lables[i] = 2
print(df)
df['lable']= new_lables
print(df)
df.to_csv('train_new.csv',index=False,header=True)
#致此，数据清洗初步完成，接下来切分为训练集和验证集
import os
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)
i = 0
os.system('cd fold5')
for train, test in kfold.split(df):
    print("%s %s" % (train, test))
    print(len(train),len(test))
    train_data = df.loc[train]
    dev_data = df.loc[test]
#     print(train_data)
    try:
        os.system('mkdir data_%d'%i)
    except:
        pass
    train_data.to_csv('data_%d/train.csv'%i, index=False, header=True)
    dev_data.to_csv('data_%d/dev.csv'%i, index=False, header=True)
    i += 1
