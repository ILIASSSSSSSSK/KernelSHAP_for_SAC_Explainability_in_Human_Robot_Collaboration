### Python scripts
* `game_control_sign.py`: Runs the game loop
* `sac_discrete_agent.py`, `networks_discrete.py`: SAC implementation
* `score_visualization.py`: Visualization of the status of the game
* `utils.py`: Functions for setting up the save directories
* `check_middle_point.py`: Used to check the position of the laser relatively to the position of the goal marker.

A GUIDE FOR THE USE OF THE KERNEL SHAP EXPLAINER
-------------------------------------------------

Note: When writing those codes in the files with the appropriate data the columns "blocks" and "episodes" are swapped (the data that should be stored as blocks are stored as episodes and vice versa) 

All the codes needed are stored in /Desktop/catkin_ws5/src/hrc_study_tsitosetal/scripts

1. histogram_duration.py:

The code when run plots a histogram which shows the frequency of games' duration.

The important part of the code is:

file_location_part_A="/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_news_100_test_experiments_"
file_names=[]
df = pd.DataFrame()
for i in range(1,66):
  if i<14:
    file_names.append(file_location_part_A+str(i)+"_14112025/data/rl_test_data.csv")
  else:
    file_names.append(file_location_part_A+str(i)+"_11032026/data/rl_test_data.csv")
    
This part finds where the games are stored. The location of the files are stored in a variable called file_names. Change this part of the code according to your preferences or just select your file_names manually after this part of the code.

2. take_only_10_22_dur.py:

This code filteres the games stored in file_names (which is defined with the same way as above). It does the following filtering:

i) First it filters all games which have durations smaller than 10 timesteps or higher than 24 timesteps (that was decided based on the histogram ploted by the code above):

blocks=range(1,521) #The total number of the games is 520 in that case. Change it so that it corresponds to your total number of games (number of games+1)
durations=[]
for i in range(len(blocks)):
  dur=0
  for j in range((len(df))):
    if df["block"].iloc[j]==blocks[i]:
      dur+=1
  durations.append(dur)
ignore_blocks=[]
for i in range(len(durations)):
  if (durations[i]<10) or (durations[i]>24): #change them based on the durations you want: here the durations of the games must be in the set [10,24]
    ignore_blocks.append(blocks[i])

ii) Then it guaranteen that the games with the same initial positions are equally distributed:

print(len(blocks_0),len(blocks_1),len(blocks_2),len(blocks_3)) #this print helps me to see the number of games with the same initial position

ignore_b0 = random.sample(blocks_0, 3) #the number of games that are needed to be removed from each initial position in order to be equally distributed 
ignore_b1 = random.sample(blocks_1, 4)
ignore_b2 = random.sample(blocks_2, 14)

iii) Then the games which have a duration with an integer 20% are used as a test set for the kernel SHAP explainer. The rest are used as the background set.

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
    
3. check_if_target_satisfied.py

file_names

["X_test_df_248_K5.pkl","test_blocks_248_K5.pkl","X_test_248_K5.csv","X_test_final_248_K5.pkl","X_test_games_248_K5.pkl"]


4.

checkpoint_path = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/rl_models/75K_every8_uniform_200ms_Ilias_news7_1_no_TL_1/actor_sac_12"
state_dict = th.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

with open("X_test_final_248_K7.pkl", "rb") as f:
    X_test=pickle.load(f)

with open("X_train_final_248_K7.pkl", "rb") as f:
    X_train=pickle.load(f)
    
np.savetxt("shap_values_try1_action_0_248_K7.csv", shap_values_action_0, delimiter=",", fmt="%f")
np.savetxt("shap_values_try1_action_1_248_K7.csv", shap_values_action_1, delimiter=",", fmt="%f")
np.savetxt("shap_values_try1_action_2_248_K7.csv", shap_values_action_2, delimiter=",", fmt="%f")

joblib.dump(explainer, "kernel_explainer_248_K7.pkl")

5.

6)
