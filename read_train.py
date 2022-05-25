import pandas as pd
from collections import Counter

# read the train.data file and transfer to csv format
df_train = pd.read_csv('./data/train.data', sep='\t', header=None)
df_train = pd.DataFrame(df_train)

df_train.columns = ['Topic_Id','Topic_Name','Sent_1','Sent_2','Label','Sent_1_tag','Sent_2_tag']

print(df_train)

df_train.info()
# RangeIndex: 13063 entries, 0 to 13062

# write the train data to csv file
df_train.to_csv("./data_preprocessing/train.csv",index=False)


# check the number that label = '(2, 3)'
# label_debatable = df_train.loc[df_train['Label'] == '(2, 3)']
# label_debatable = pd.DataFrame(label_debatable)
# label_debatable.info()
#Int64Index: 1533 entries, 3 to 13051

#drop the lines that Label = '(2, 3)'
df_train_new = df_train.query("Label != '(2, 3)'")
df_train_new = df_train_new.reset_index()
df_train_new.drop(["index"],axis=1, inplace=True)

# write the train data to csv file, do not include lines that Label == '(2, 3)'
df_train_new.to_csv("./data_preprocessing/train_01_ori.csv",index=False)

# transfer the value of label to 0 or 1
df_train_01 = df_train_new.copy()
df_train_01['Label'].replace('(0, 5)',0, inplace=True)
df_train_01['Label'].replace('(1, 4)',0, inplace=True)

df_train_01['Label'].replace('(3, 2)',1, inplace=True)
df_train_01['Label'].replace('(4, 1)',1, inplace=True)
df_train_01['Label'].replace('(5, 0)',1, inplace=True)

Counter(df_train_01['Label'])
#Counter({1: 3996, 0: 7534})

# write the train data to csv file, Label = 0 or 1
df_train_01.to_csv("./data_preprocessing/train_01.csv",index=False)