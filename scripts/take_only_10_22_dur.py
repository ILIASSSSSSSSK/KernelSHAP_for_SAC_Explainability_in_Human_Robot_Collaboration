import pickle
import numpy as np
import math as mt
import pandas as pd 
import random

file_location_part_A="/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_news_100_test_experiments_"
file_names=[]
df = pd.DataFrame()
for i in range(1,66):
  if i<14:
    file_names.append(file_location_part_A+str(i)+"_14112025/data/rl_test_data.csv")
  else:
    file_names.append(file_location_part_A+str(i)+"_11032026/data/rl_test_data.csv")
for i in range(len(file_names)):
  df1=pd.read_csv(file_names[i])
  for j in range(len(df1)):
    if df1["block"][j]!=0:
      #αλλάζω την τιμή του block γιατί κάθε αρχείο έχει από 1-8 blocks και θέλω κάθε block να έχει το δικό του id number. Άρα αντί να έχω 13 διαφορετικά 1-8, τα κάνω ώστε να είναι από 1-104 προσθέτωντας κάθε φορά ID αρχείου*8
      df1["block"][j]=df1["block"][j]+i*8
  df=pd.concat([df,df1])

blocks=range(1,521)
durations=[]
for i in range(len(blocks)):
  dur=0
  for j in range((len(df))):
    if df["block"].iloc[j]==blocks[i]:
      dur+=1
  durations.append(dur)
ignore_blocks=[]
for i in range(len(durations)):
  if (durations[i]<10) or (durations[i]>24):
    ignore_blocks.append(blocks[i])

final_blocks=[]
for i in range(len(blocks)):
  if blocks[i] not in ignore_blocks:
    final_blocks.append(blocks[i])

print(len(final_blocks))

pos_0=0
pos_1=0
pos_2=0
pos_3=0
blocks_0=[]
blocks_1=[]
blocks_2=[]
blocks_3=[]
for i in range(len(final_blocks)):
  for j in range((len(df))):
    if df["block"].iloc[j]==final_blocks[i]:
      if df["init_pos"].iloc[j]==0:
        pos_0+=1
        blocks_0.append(final_blocks[i])
      elif df["init_pos"].iloc[j]==1:
        pos_1+=1
        blocks_1.append(final_blocks[i])
      elif df["init_pos"].iloc[j]==2:
        pos_2+=1
        blocks_2.append(final_blocks[i])
      elif df["init_pos"].iloc[j]==3:
        pos_3+=1
        blocks_3.append(final_blocks[i])
      break
print(len(blocks_0),len(blocks_1),len(blocks_2),len(blocks_3))

ignore_b0 = random.sample(blocks_0, 3)
ignore_b1 = random.sample(blocks_1, 4)
ignore_b2 = random.sample(blocks_2, 14)

blocks_0_f=[]
blocks_1_f=[]
blocks_2_f=[]
for i in blocks_0:
  if i not in ignore_b0:
    blocks_0_f.append(i)
for i in blocks_1:
  if i not in ignore_b1:
    blocks_1_f.append(i)
for i in blocks_2:
  if i not in ignore_b2:
    blocks_2_f.append(i)
print(len(blocks_0_f),len(blocks_1_f),len(blocks_2_f),len(blocks_3))

def filtered_durations(blocks):
  blocks_with_durations_int_20_perc=[]
  the_other_blocks=[]
  for i in range(len(blocks)):
    dur=0
    for j in range((len(df))):
      if df["block"].iloc[j]==blocks[i]:
        dur+=1
    if (dur*0.2).is_integer():
      blocks_with_durations_int_20_perc.append(blocks[i])
    else:
      the_other_blocks.append(blocks[i])
  return blocks_with_durations_int_20_perc,the_other_blocks

blocks_0_20_per,blocks_0_not20=filtered_durations(blocks_0_f)
blocks_1_20_per,blocks_1_not20=filtered_durations(blocks_1_f)
blocks_2_20_per,blocks_2_not20=filtered_durations(blocks_2_f)
blocks_3_20_per,blocks_3_not20=filtered_durations(blocks_3)
print(len(blocks_0_20_per),len(blocks_0_not20))
print(len(blocks_1_20_per),len(blocks_1_not20))
print(len(blocks_2_20_per),len(blocks_2_not20))
print(len(blocks_3_20_per),len(blocks_3_not20))

min_20=min([len(blocks_0_20_per),len(blocks_1_20_per),len(blocks_2_20_per),len(blocks_3_20_per)])
print(min_20)
"""
blocks_0_train=[]
blocks_1_train=[]
blocks_2_train=[]
blocks_3_train=[]
train_pos=62
blocks_0_train=random.sample(blocks_0_f, 62)
blocks_1_train=random.sample(blocks_1_f, 62)
blocks_2_train=random.sample(blocks_2_f, 62)
blocks_3_train=random.sample(blocks_3, 62)
bt=[blocks_0_train,blocks_1_train,blocks_2_train,blocks_3_train]
train_blocks=[]
for i in bt:
  for j in range(len(i)):
    train_blocks.append(i[j])
print(len(train_blocks))
blocks_final=[blocks_0_f,blocks_1_f,blocks_2_f,blocks_3]
test_blocks=[]
for i in blocks_final:
  for j in range(len(i)):
    if i[j] not in train_blocks:
      test_blocks.append(i[j])
print(len(test_blocks))
"""
blocks_0_test=[]
blocks_1_test=[]
blocks_2_test=[]
blocks_3_test=[]
blocks_0_test=random.sample(blocks_0_20_per, min_20)
blocks_1_test=random.sample(blocks_1_20_per, min_20)
blocks_2_test=random.sample(blocks_2_20_per, min_20)
blocks_3_test=random.sample(blocks_3_20_per, min_20)
bt=[blocks_0_test,blocks_1_test,blocks_2_test,blocks_3_test]
test_blocks=[]
for i in bt:
  for j in range(len(i)):
    test_blocks.append(i[j])

blocks_final=[blocks_0_f,blocks_1_f,blocks_2_f,blocks_3]
train_blocks=[]
for i in blocks_final:
  for j in range(len(i)):
    if i[j] not in test_blocks:
      train_blocks.append(i[j])
print(len(train_blocks))
print(len(test_blocks))


#-------------------------------------------------------------------------------------------------------
X_train_df=[]
X_test_df=[]
state_f = ['ee_pos_x_prev', 'ee_pos_y_prev', 'ee_vel_x_prev', 'ee_vel_y_prev','block','init_pos']

for i in range(len(df)):
    #αν η γραμμή στο df έχει block id που είναι στο train βάλε το στο training set
    if df["block"].iloc[i] in train_blocks:
        X_train_df.append(df[state_f].iloc[i])
    #αλλιώς αν δεν είναι 0 στο test set (αν είναι 0 είναι αλλαγή μεταξύ παιχιδιών άρα άχρηστο)
    elif df["block"].iloc[i] in test_blocks:
      if df["block"].iloc[i]!=0:
        X_test_df.append(df[state_f].iloc[i])

#κάνε τις λίστες πίνακα
X_test_all=np.array(X_test_df) #all info that i need for graphs
X_train_all=np.array(X_train_df)
#take only features for shap (4 feature states)
X_test=X_test_all[:,:4]
X_train=X_train_all[:,:4]


np.savetxt("X_test_248_K3.csv",X_test,delimiter=",", fmt="%f")

with open("X_test_df_248_K3.pkl", "wb") as f:
    pickle.dump(X_test_df, f)
with open("test_blocks_248_K3.pkl", "wb") as f:
    pickle.dump(test_blocks, f)

np.savetxt("X_train_248_K3.csv",X_train,delimiter=",", fmt="%f")
with open("X_train_df_248_K3.pkl", "wb") as f:
    pickle.dump(X_train_df, f)
with open("train_blocks_248_K3.pkl", "wb") as f:
    pickle.dump(train_blocks, f)