import shap
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

shap_values_0 = np.loadtxt("shap_values_try1_action_0_248_K3.csv", delimiter=",")
shap_values_1 = np.loadtxt("shap_values_try1_action_1_248_K3.csv", delimiter=",")
shap_values_2 = np.loadtxt("shap_values_try1_action_2_248_K3.csv", delimiter=",")
shap_values = np.stack((shap_values_0, shap_values_1, shap_values_2), axis=2)

with open("X_test_final_248_K3.pkl", "rb") as f:
    X_test=pickle.load(f)
with open("X_test_games_248_K3.pkl",'rb') as file:
  X_test_games=pickle.load(file)

goal_pos=[-0.252, 0.245]
max_x= -0.162
min_x= -0.350
max_y= 0.348
min_y= 0.159
#ΠΡΟΣΟΧΗ: ΔΕΝ ΗΤΑΝ ΑΡΧΙΚΕΣ ΘΕΣΕΙΣ ΑΛΛΑ ΤΕΤΑΡΤΗΜΟΡΙΑ. ΑΛΛΑΞΕ ΟΝΟΜΑΤΑ ΑΡΓΟΤΕΡΑ
init_pos_games=[]
for i in X_test_games:
    init_pos_one_game=[]
    for j in range(len(i)):
        if ((i[j][0]*(max_x-min_x)+min_x)>goal_pos[0]) and ((i[j][1]*(max_y-min_y)+min_y)>goal_pos[1]):
            init_pos_one_game.append(0)
        elif ((i[j][0]*(max_x-min_x)+min_x)<goal_pos[0]) and ((i[j][1]*(max_y-min_y)+min_y)>goal_pos[1]): 
            init_pos_one_game.append(1)
        elif ((i[j][0]*(max_x-min_x)+min_x)<goal_pos[0]) and ((i[j][1]*(max_y-min_y)+min_y)<goal_pos[1]):
            init_pos_one_game.append(2)
        elif ((i[j][0]*(max_x-min_x)+min_x)>goal_pos[0]) and ((i[j][1]*(max_y-min_y)+min_y)<goal_pos[1]):
            init_pos_one_game.append(3)
    init_pos_games.append(init_pos_one_game)
init_pos_test=[]
for i in init_pos_games:
    for j in range(len(i)):
        init_pos_test.append(i[j])

def organize_in_positions(pos):
    shap_pos=[]
    X_pos=[]
    for i in range(len(X_test)):
        if init_pos_test[i]==pos:
            shap_pos.append(shap_values[i])
            X_pos.append(X_test[i])
    return X_pos, shap_pos

X_test_pos_0,shap_test_pos_0=organize_in_positions(0)
X_test_pos_1,shap_test_pos_1=organize_in_positions(1)
X_test_pos_2,shap_test_pos_2=organize_in_positions(2)
X_test_pos_3,shap_test_pos_3=organize_in_positions(3)

#πώς κατανέμοντα οι shap values σε όλο το test set, σε κάθε feauture για το action 0,1 και 2
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
        greedy_actions = th.argmax(actions_logits, dim=-1, keepdim=True) #take only max action
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
def plot_beeswarm(X_test,shap_values,action,cmap,alpha,title,only_selected):
 if only_selected==True:
    X_test_new=[]
    shap_test_new=[]
    for i in range(len(X_test)):
        if model_predict(X_test[i])==action:
            X_test_new.append(X_test[i])
            shap_test_new.append(shap_values[i])
    X_test=np.array(X_test_new)
    shap_values=np.array(shap_test_new)
 shap.summary_plot(X_test,shap_values[:, :, action], feature_names=["x","y","u_x","u_y"], show=False,cmap=cmap,color_bar=False,sort=False,alpha=alpha)
 
 plt.title(title, fontsize=8)
 
 ax = plt.gca()
 feature_values = [(0.05+0.2)/0.4, (0.05+0.2)/0.4, (goal_pos[1]+0.01-min_y)/(max_y-min_y), (goal_pos[0]+0.01-min_x)/(max_x-min_x)]
 for i, val in enumerate(feature_values):
    ax.plot(
        [val, val],          # x = constant → vertical
        [i - 0.4, i + 0.4],  # small segment around feature row
        linestyle='--',
        color='black',
        linewidth=1,
        zorder=10)
 feature_values = [(-0.05+0.2)/0.4, (-0.05+0.2)/0.4, (goal_pos[1]-0.01-min_y)/(max_y-min_y), (goal_pos[0]-0.01-min_x)/(max_x-min_x)]
 for i, val in enumerate(feature_values):
    ax.plot(
        [val, val],          # x = constant → vertical
        [i - 0.4, i + 0.4],  # small segment around feature row
        linestyle='--',
        color='black',
        linewidth=1,
        zorder=10)
 ax.set_facecolor("#f5f5f5")
 ax.set_xlabel("Feature normalized value", fontsize=7)
 ax.plot([], [], linestyle='--', color='black', linewidth=1, label='Satisfaction of goal state')
 ax.legend(loc='lower left', fontsize=5)
 

 for c in ax.collections:
     c.set_clim(-0.64,0.64)
 mappable = ax.collections[3]

 mappable.set_clim(-0.64,0.64)
 # Axis label sizes
 ax.set_xlabel(ax.get_xlabel(), fontsize=7)
 ax.set_ylabel(ax.get_ylabel(), fontsize=7)

 # Tick sizes
 ax.tick_params(axis='both', labelsize=7)

 # Y labels (feature names)
 ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
 cbar = plt.colorbar(mappable)
 cbar.set_label("SHAP value", fontsize=7)
 cbar.ax.tick_params(labelsize=7)
 plt.xlim([0,1])

cmaps=["coolwarm",'bwr', 'seismic','rainbow']

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),0,cmaps[0],1,"quartile 0",False)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),0,cmaps[1],1,"quartile 1",False)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),0,cmaps[2],1,"quartile 2",False)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),0,cmaps[3],1,"quartile 3",False)
plt.suptitle("Contribution of each feature in all observations for action 0")
plt.show()

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),1,cmaps[0],1,"quartile 0",False)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),1,cmaps[1],1,"quartile 1",False)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),1,cmaps[2],1,"quartile 2",False)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),1,cmaps[3],1,"quartile 3",False)
plt.suptitle("Contribution of each feature in all observations for action 1")
plt.show()

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),2,cmaps[0],1,"quartile 0",False)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),2,cmaps[1],1,"quartile 1",False)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),2,cmaps[2],1,"quartile 2",False)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),2,cmaps[3],1,"quartile 3",False)
plt.suptitle("Contribution of each feature in all observations for action -1")
plt.show()

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),0,cmaps[0],1,"quartile 0",True)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),0,cmaps[1],1,"quartile 1",True)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),0,cmaps[2],1,"quartile 2",True)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),0,cmaps[3],1,"quartile 3",True)
plt.suptitle("Contribution of each feature in all observations for action 0")
plt.show()

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),1,cmaps[0],1,"quartile 0",True)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),1,cmaps[1],1,"quartile 1",True)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),1,cmaps[2],1,"quartile 2",True)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),1,cmaps[3],1,"quartile 3",True)
plt.suptitle("Contribution of each feature in all observations for action 1")
plt.show()

plt.figure(figsize=(2, 0.8),dpi=200)
plt.subplot(2, 2, 1)
plot_beeswarm(np.array(X_test_pos_0),np.array(shap_test_pos_0),2,cmaps[0],1,"quartile 0",True)
plt.subplot(2, 2, 2)
plot_beeswarm(np.array(X_test_pos_1),np.array(shap_test_pos_1),2,cmaps[1],1,"quartile 1",True)
plt.subplot(2, 2, 3)
plot_beeswarm(np.array(X_test_pos_2),np.array(shap_test_pos_2),2,cmaps[2],1,"quartile 2",True)
plt.subplot(2, 2, 4)
plot_beeswarm(np.array(X_test_pos_3),np.array(shap_test_pos_3),2,cmaps[3],1,"quartile 3",True)
plt.suptitle("Contribution of each feature in all observations for action -1")
plt.show()