import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

duration=[]

for i in range(1,521):
	dur=0
	for j in range(len(df)):
		if df["block"].iloc[j]==i:
			dur+=1
	duration.append(dur)


# Histogram
sns.histplot(duration, bins=50, kde = True)

plt.xlabel("Duration in Timesteps", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.title("Histogram of games' duration", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

std = np.std(duration)
mean=np.mean(duration)
min_dur = np.min(duration)
max_dur = np.max(duration)
Qs=np.percentile(duration, [25, 50, 75])
print("Max duration: ",max_dur)
print("Min duration: ",min_dur)
print("Mean duration: " ,mean)
print("Std: ",std)
print("Q1: ",Qs[0])
print("Q2: ",Qs[1])
print("Q3: ",Qs[2])