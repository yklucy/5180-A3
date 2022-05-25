import pandas as pd
from collections import Counter

# read the test.data file and transfer to csv format
df_test = pd.read_csv('./data/test.data', sep='\t', header=None)
df_test = pd.DataFrame(df_test)

df_test.columns = ['Topic_Id','Topic_Name','Sent_1','Sent_2','Label','Sent_1_tag','Sent_2_tag']

print(df_test)

df_test.info()
# RangeIndex: 972 entries, 0 to 971

# write the test data to csv file
df_test.to_csv("./data_preprocessing/test.csv",index=False)


# check the number that label = 3
# label_debatable = df_test.loc[df_test['Label'] == 3]
# label_debatable = pd.DataFrame(label_debatable)
# label_debatable.info()
# Int64Index: 134 entries, 0 to 952

#drop the lines that Label = 3
df_test_new = df_test.query("Label != 3")
df_test_new = df_test_new.reset_index()
df_test_new.drop(["index"],axis=1, inplace=True)

# write the test data to csv file, do not include lines that Label == 3
df_test_new.to_csv("./data_preprocessing/test_01_ori.csv",index=False)

# transfer the value of label to 0 or 1
df_test_01 = df_test_new.copy()
df_test_01['Label'].replace(1, 0, inplace=True)
df_test_01['Label'].replace(2, 0, inplace=True)

df_test_01['Label'].replace(4, 1, inplace=True)
df_test_01['Label'].replace(5, 1, inplace=True)

Counter(df_test_01['Label'])
#Counter({0: 663, 1: 175})

# write the test data to csv file, Label = 0 or 1
df_test_01.to_csv("./data_preprocessing/test_01.csv",index=False)