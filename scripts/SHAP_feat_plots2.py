import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.patches as mpatches
import math as mt
import joblib

shap_values_0 = np.loadtxt("shap_values_try1_action_0_248_K4.csv", delimiter=",")
shap_values_1 = np.loadtxt("shap_values_try1_action_1_248_K4.csv", delimiter=",")
shap_values_2 = np.loadtxt("shap_values_try1_action_2_248_K4.csv", delimiter=",")

with open("X_test_games_248_K4.pkl",'rb') as file:
  X_test_games=pickle.load(file)

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units):
        super(Actor, self).__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, action_dim)
        )

    def forward(self, s):
        
        #Forward pass: returns action logits and greedy actions
        
        actions_logits = self.actor_mlp(s)
        greedy_actions = actions_logits
        #κράτα το με softmax (έτσι συμβαίνει και σε βιβλιογραφία)->εξηγείς τον τρόπο με τον οποίο κατανεμήθηκαν οι πιθανότητες στα actions
        greedy_actions= F.softmax(actions_logits, dim=-1) #produce possibilities for the three possible actions in order to have a view and for the three of them
        #greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True) #take only max action
        return greedy_actions


#Step 1: Όρισε τις παραμέτρους του δικτύου
state_dim = 4       # dimension of state vector
action_dim = 3       # number of actions
n_hidden_units = 32 # hidden layer size

model = Actor(state_dim, action_dim, n_hidden_units).to(device)

# Step 2: Load the saved parameters (state_dict)
checkpoint_path = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/rl_models/75K_every8_uniform_200ms_Ilias_news7_1_no_TL_1/actor_sac_12"
state_dict = th.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

# Step 3: Set model to evaluation mode (γίνεται εδώ σε evaluation mode άρα δεχ ρειάζεται να το ξαναβάλω στο predict function)
model.eval()

# @ Define a prediction function for generating actions for SHAP Explainer
def model_predict(X):
    X_tensor = th.tensor(X, dtype=th.float32)
    X_tensor=X_tensor.cuda()
    model.eval()
    with th.no_grad():
        return model(X_tensor).detach().cpu().numpy()
explainer = joblib.load("kernel_explainer_248_K3.pkl")

shap_values_games_all=[]
im=0
selected_actions_games=[]

X_test_games_pos0=[]
X_test_games_pos1=[]
X_test_games_pos2=[]
X_test_games_pos3=[]

shap_games_pos0=[]
shap_games_pos1=[]
shap_games_pos2=[]
shap_games_pos3=[]

goal_pos=[-0.252, 0.245]
max_x= -0.162
min_x= -0.350
max_y= 0.348
min_y= 0.159

for i in range(len(X_test_games)):
  shap_one_game=[]
  selected_action_one_game=[]
  for j in range(len(X_test_games[i])):
    selected_action=model_predict(X_test_games[i][j]).argmax()
    selected_action_one_game.append(selected_action)
    if selected_action==0:
      shap_one_game.append(shap_values_0[im+j])
    elif selected_action==1:
      shap_one_game.append(shap_values_1[im+j])
    elif selected_action==2:
      shap_one_game.append(shap_values_2[im+j])
  if ((X_test_games[i][0][0]*(max_x-min_x)+min_x)>goal_pos[0]) and ((X_test_games[i][0][1]*(max_y-min_y)+min_y)>goal_pos[1]):
    X_test_games_pos0.append(X_test_games[i])
    shap_games_pos0.append(shap_one_game)
  elif ((X_test_games[i][0][0]*(max_x-min_x)+min_x)<goal_pos[0]) and ((X_test_games[i][0][1]*(max_y-min_y)+min_y)>goal_pos[1]):
    X_test_games_pos1.append(X_test_games[i])
    shap_games_pos1.append(shap_one_game)
  elif ((X_test_games[i][0][0]*(max_x-min_x)+min_x)<goal_pos[0]) and ((X_test_games[i][0][1]*(max_y-min_y)+min_y)<goal_pos[1]):
    X_test_games_pos3.append(X_test_games[i])
    shap_games_pos3.append(shap_one_game)
  elif ((X_test_games[i][0][0]*(max_x-min_x)+min_x)>goal_pos[0]) and ((X_test_games[i][0][1]*(max_y-min_y)+min_y)<goal_pos[1]):
    X_test_games_pos2.append(X_test_games[i])
    shap_games_pos2.append(shap_one_game)
  shap_values_games_all.append(shap_one_game)
  selected_actions_games.append(selected_action_one_game)
  im+=len(X_test_games[i])
print(len(X_test_games_pos0)==len(X_test_games_pos1)==len(X_test_games_pos2)==len(X_test_games_pos3))
print(len(X_test_games_pos0))
print(len(shap_games_pos3)==len(shap_games_pos1)==len(shap_games_pos2)==len(shap_games_pos3))
print(len(shap_games_pos0))
print((len(shap_games_pos0)*4)==len(shap_values_games_all))

def plot_shap(shap_values_games_all,colors,labels,changed,target,title,ylim,ylabel=""):
 shap_features=[]
 pos_dist=[-2.5,0,2.5,5]
 for j in range(len(shap_values_games_all[0][0])):
  shap_0=[]
  shap_20=[]
  shap_40=[]
  shap_60=[]
  shap_80=[]
  shap_100=[]
  for i in range(len(shap_values_games_all)):
   t_0=0
   t_20=mt.floor(len(shap_values_games_all[i])*0.2)-1
   t_40=mt.floor(len(shap_values_games_all[i])*0.4)-1
   t_60=mt.floor(len(shap_values_games_all[i])*0.6)-1
   t_80=mt.floor(len(shap_values_games_all[i])*0.8)-1
   t_100=len(shap_values_games_all[i])-1
   if changed==True:
     shap_0.append(shap_values_games_all[i][t_0][j])
     shap_20.append(shap_values_games_all[i][t_20][j])
     shap_40.append(shap_values_games_all[i][t_40][j])
     shap_60.append(shap_values_games_all[i][t_60][j])
     shap_80.append(shap_values_games_all[i][t_80][j])
     shap_100.append(shap_values_games_all[i][t_100][j])
   else:
     T = len(shap_values_games_all[i])
     s0=[]
     s20=[]
     s40=[]
     s60=[]
     s80=[]
     s100=[]
     dt=mt.floor(len(shap_values_games_all[i])*0.1)
     for t in range(max(0, t_0 - dt), min(T, t_0 + dt + 1)):
      s0.append(shap_values_games_all[i][t][j])
     for t in range(max(0, t_20 - dt), min(T, t_20 + dt + 1)):
      s20.append(shap_values_games_all[i][t][j])
     for t in range(max(0, t_40 - dt), min(T, t_40 + dt + 1)):
      s40.append(shap_values_games_all[i][t][j])
     for t in range(max(0, t_60 - dt), min(T, t_60 + dt + 1)):
      s60.append(shap_values_games_all[i][t][j])
     for t in range(max(0, t_80 - dt), min(T, t_80 + dt + 1)):
      s80.append(shap_values_games_all[i][t][j])
     for t in range(max(0, t_100 - dt), min(T, t_100 + dt + 1)):
      s100.append(shap_values_games_all[i][t][j])
     shap_0.append(np.mean(s0))
     shap_20.append(np.mean(s20))
     shap_40.append(np.mean(s40))
     shap_60.append(np.mean(s60))
     shap_80.append(np.mean(s80))
     shap_100.append(np.mean(s100))

  box=plt.boxplot(
    #[shap_0, shap_20, shap_40, shap_60, shap_80, shap_100],
    #positions=[0+pos_dist[j], 20+pos_dist[j], 40+pos_dist[j], 60+pos_dist[j], 80+pos_dist[j], 100+pos_dist[j]],
    [shap_20, shap_40, shap_60, shap_80, shap_100],
    positions=[20+pos_dist[j], 40+pos_dist[j], 60+pos_dist[j], 80+pos_dist[j], 100+pos_dist[j]],
    widths=2,
    patch_artist=True,
    showfliers=True,

    boxprops=dict(facecolor=colors[j], color=colors[j]),
    medianprops=dict(color=colors[j],linewidth=2),
    whiskerprops=dict(color=colors[j]),
    capprops=dict(color=colors[j]),
    flierprops=dict(marker='o', markerfacecolor=colors[j], markeredgecolor=colors[j]))
  for patch in box['boxes']:
    patch.set(facecolor=colors[j],alpha=0.5)
  #plt.xticks([0,20,40,60,80,100],["0%","20%","40%","60%","80%","100%"])
  plt.xticks([20,40,60,80,100],["20%","40%","60%","80%","100%"],fontsize=15)
  plt.yticks(fontsize=12)
 plt.xlabel("Percentages (%)",fontsize=15)
 # ---- Create custom legend ----
 legend_patches=[]
 if len(labels)==4:
  legend_patches = [
    mpatches.Patch(facecolor=colors[0], alpha=0.3, label=labels[0]),
    mpatches.Patch(facecolor=colors[1], alpha=0.3, label=labels[1]),
    mpatches.Patch(facecolor=colors[2], alpha=0.3, label=labels[2]),
    mpatches.Patch(facecolor=colors[3], alpha=0.3, label=labels[3])]
  plt.ylim([-0.4,0.65])
  plt.ylabel("SHAP",fontsize=15)
  plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)
 else:
  legend_patches = [
    mpatches.Patch(facecolor=colors[0], alpha=0.3, label=labels[0]),
    mpatches.Patch(facecolor=colors[1], alpha=0.3, label=labels[1]),
    mpatches.Patch(facecolor=colors[2], alpha=0.3, label=labels[2])]
  plt.axhline(y=target, color='black', linestyle='--', linewidth=0.7)
  plt.ylim([0,ylim])
  plt.ylabel(ylabel,fontsize=15)



 # Place legend outside the plot
 plt.legend(handles=legend_patches, title='Legend', loc='upper right',
           bbox_to_anchor=(1, 1),fontsize=14,title_fontsize=15)  # (x, y) relative to axes

 # Adjust layout to make room for the legend
 #plt.tight_layout(rect=[0, 0, 0.75, 1])  # leave space on right
 plt.title(title, fontsize=17)
 plt.show()

def plot_features(X_test_games,title):
  X_test_games_unnormalize_x_y_dist=[]
  X_test_games_unnormalize_velx_vely_vel=[]
  max_x= -0.162
  min_x= -0.350
  max_y= 0.348
  min_y= 0.159
  goal_pos=[-0.252, 0.245]
  min_vel=-0.2
  max_vel=0.2
  for i in X_test_games:
    X_one_game1=[]
    X_one_game2=[]
    for j in range(len(i)):
      x=abs(i[j][0]*(max_x-min_x)+min_x-goal_pos[0])
      y=abs(i[j][1]*(max_y-min_y)+min_y-goal_pos[1])
      vel_x=abs(i[j][2]*(max_vel-min_vel)+min_vel)
      vel_y=abs(i[j][3]*(max_vel-min_vel)+min_vel)
      vel_t=mt.sqrt(vel_x**2+vel_y**2)
      dist=mt.sqrt(x**2+y**2)
      X_one_game1.append([x,y,dist])
      X_one_game2.append([vel_x,vel_y,vel_t])
    X_test_games_unnormalize_x_y_dist.append(X_one_game1)
    X_test_games_unnormalize_velx_vely_vel.append(X_one_game2)
  plot_shap(X_test_games_unnormalize_x_y_dist,["red","blue","grey"],["|x-x_target|","|y-y_target|","Distance from target"],True,0.01,"EE's distance from target in each percentage of motion "+title,0.14,"Distance (m)")
  plot_shap(X_test_games_unnormalize_velx_vely_vel,["orange","cyan","grey"],["|u_x|","|u_y|","|u|"],True,0.05,"EE's speed in each percentage of motion "+title,0.15,"Speed (m/s)")


plot_shap(shap_values_games_all,["red","blue","orange","cyan"],["x","y","u_x","u_y"],True,0,"SHAP values in each percentage of motion for all Positions",0)
plot_features(X_test_games,"for all Positions")

plot_shap(shap_games_pos0,["red","blue","orange","cyan"],["x","y","u_x","u_y"],True,0,"SHAP values in each percentage of motion for all games with initial position UR",0)
plot_features(X_test_games_pos0,"for games with initial Position UR")

plot_shap(shap_games_pos1,["red","blue","orange","cyan"],["x","y","u_x","u_y"],True,0,"SHAP values in each percentage of motion for all games with initial position UL",0)
plot_features(X_test_games_pos1,"for games with initial Position UL")

plot_shap(shap_games_pos2,["red","blue","orange","cyan"],["x","y","u_x","u_y"],True,0,"SHAP values in each percentage of motion for all games with initial position LR",0)
plot_features(X_test_games_pos2,"for games with initial Position LR")

plot_shap(shap_games_pos3,["red","blue","orange","cyan"],["x","y","u_x","u_y"],True,0,"SHAP values in each percentage of motion for all games with initial position LL",0)
plot_features(X_test_games_pos3,"for games with initial Position LL")