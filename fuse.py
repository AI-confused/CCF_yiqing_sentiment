import pandas as pd
import os
import numpy as np

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)  
files = []
ROOT = './logit_result'

listdir(ROOT, files)
print(files)
result = pd.read_csv(files[0])
for file in files[1:]:
    result.iloc[:,1:] += pd.read_csv(file).iloc[:,1:]

result['y'] = np.argmax(result.iloc[:,1:].values, 1) - 1 # -1,0,1
df = pd.read_csv('data/test_new.csv')
na_df = list(df[df['flag']==1].index)
result.iloc[na_df,-1] = 0
result[['id', 'y']].to_csv('./result.csv', index=False, header=True)
print('model fuse done')