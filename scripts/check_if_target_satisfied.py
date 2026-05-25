import pickle
import numpy as np
import math as mt
import pandas as pd 

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

#print(df)
#change "test" with "train" to correct that to
files_to_open_and_store=["X_test_df_248_K5.pkl","test_blocks_248_K5.pkl","X_test_248_K5.csv","X_test_final_248_K5.pkl","X_test_games_248_K5.pkl"]
with open(files_to_open_and_store[0],'rb') as file:
  X_test_df=pickle.load(file)
with open(files_to_open_and_store[1],'rb') as file:
  test_blocks=pickle.load(file)
X_test=np.loadtxt(files_to_open_and_store[2],delimiter=",")
X_test_games=[]

max_x= -0.162
min_x= -0.350
max_y= 0.348
min_y= 0.159
goal_pos=[-0.252, 0.245]
min_vel=-0.2
max_vel=0.2
#print(test_blocks)
#print(X_test_df)

for b in range(len(test_blocks)):
  X_one_game=[]
  for i in range(len(X_test_df)):
      if X_test_df[i]["block"]==test_blocks[b]:
        X_one_game.append(X_test[i])

  
  X_one_game.pop(0)        
  X_test_games.append(X_one_game)

X_next_last=[]
for b in range(len(test_blocks)):
  X_next=[]
  for i in range(len(df)):
    if df["block"].iloc[i]==test_blocks[b]:
      X_next.append([df["ee_pos_x_next"].iloc[i],df["ee_pos_y_next"].iloc[i],df["ee_vel_x_next"].iloc[i],df["ee_vel_y_next"].iloc[i]])
  X_next_last.append(X_next[len(X_next)-1])

for i in range(len(X_test_games)):
  X_test_games[i].append(X_next_last[i])
print(X_test_games[len(X_test_games)-1])
for i in range(len(X_test_games)):
  x=X_test_games[i][len(X_test_games[i])-1][0]*(max_x-min_x)+min_x
  y=X_test_games[i][len(X_test_games[i])-1][1]*(max_y-min_y)+min_y
  vel_x=X_test_games[i][len(X_test_games[i])-1][2]*(max_vel-min_vel)+min_vel
  vel_y=X_test_games[i][len(X_test_games[i])-1][3]*(max_vel-min_vel)+min_vel
  d=mt.sqrt((x-goal_pos[0])**2+(y-goal_pos[1])**2)
  vel=mt.sqrt(vel_x**2+vel_y**2)
  if (d>0.01):
    print("Target not satisfied: ",d," Block: ",test_blocks[i]," x: ",x," y: ",y)
  if (vel>0.05):
    print("Target not satisfied")
pressed_key=input("Would you like to save the new data? Y/N? ")
if pressed_key=="Y":
  print("Saving data")
  X_test_final=[]
  for i in X_test_games:
    for j in range(len(i)):
      X_test_final.append(i[j])
  X_test_final=np.array(X_test_final)
  print(X_test_final)
  
  with open(files_to_open_and_store[3], "wb") as f:
    pickle.dump(X_test_final, f)
  with open(files_to_open_and_store[4],"wb") as f:
    pickle.dump(X_test_games,f)
  
else:
  print("Ok bye")
