import pandas as pd
from collections import Counter

# read the dev.data file and transfer to csv format
df_dev = pd.read_csv('./data/dev.data', sep='\t', header=None)
df_dev = pd.DataFrame(df_dev)

df_dev.columns = ['Topic_Id','Topic_Name','Sent_1','Sent_2','Label','Sent_1_tag','Sent_2_tag']

print(df_dev)

df_dev.info()
# RangeIndex: 4727 entries, 0 to 4726 

# write the dev data to csv file
df_dev.to_csv("./data_preprocessing/dev.csv",index=False)

# check the number that label = '(2,3)'
# label_debatable = df_dev.loc[df_dev['Label'] == '(2, 3)']
# label_debatable = pd.DataFrame(label_debatable)
# label_debatable.info()
# Int64Index: 585 entries, 6 to 4715

#drop the lines that Label = '(2, 3)'
df_dev_new = df_dev.query("Label != '(2, 3)'")
df_dev_new = df_dev_new.reset_index()
df_dev_new.drop(["index"],axis=1, inplace=True)

# write the dev data to csv file, do not include lines that Label == '(2, 3)'
df_dev_new.to_csv("./data_preprocessing/dev_01_ori.csv",index=False)

# transfer the value of label to 0 or 1
df_dev_01 = df_dev_new.copy()
df_dev_01['Label'].replace('(0, 5)',0, inplace=True)
df_dev_01['Label'].replace('(1, 4)',0, inplace=True)

df_dev_01['Label'].replace('(3, 2)',1, inplace=True)
df_dev_01['Label'].replace('(4, 1)',1, inplace=True)
df_dev_01['Label'].replace('(5, 0)',1, inplace=True)

Counter(df_dev_01['Label'])
#Counter({0: 2672, 1: 1470})

# write the dev data to csv file, Label = 0 or 1
df_dev_01.to_csv("./data_preprocessing/dev_01.csv",index=False)