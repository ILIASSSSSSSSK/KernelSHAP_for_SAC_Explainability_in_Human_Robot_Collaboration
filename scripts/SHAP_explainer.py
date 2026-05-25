import os
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import shap
from torchsummary import summary
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import seaborn as sns
from dtaidistance import dtw
from scipy.stats import pearsonr
import pickle
import joblib

#-------------------------------------------------------------------------------------------------------

#many files
#δημιουργώ ένα νέο df που περιλαμβάνει όλα τα test παιχνίδια από τα 13 αρχεία που έχω διαθέσιμα
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

 #-------------------------------------------------------------------------------------------------------

 #όρισε το δίκτυο του Actor ώστε να είναι το ίδιο με τον actor του παιχνιδιού
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
        """
        Forward pass: returns action logits and greedy actions
        """
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

summary(model, input_size=(state_dim,))

with open("X_test_final_248_K7.pkl", "rb") as f:
    X_test=pickle.load(f)

with open("X_train_final_248_K7.pkl", "rb") as f:
    X_train=pickle.load(f)

"""
#όλο αυτό πλέον δε χρειάζεται αφού γίνεται σε άλλο αρχείο κώδικα
#-------------------------------------------------------------------------------------------------------

#for many files
#δημιουργώ υποσύνολο που να έχει μόνο pos_x_prev,pos_y_prev,vel_x_prev,vel_y_prev, block και init_pos (prev=η τωρινή κατάσταση)
#δημιουργία train και test set για τον explainer
X_train_df=[]
X_test_df=[]
state_f = ['ee_pos_x_prev', 'ee_pos_y_prev', 'ee_vel_x_prev', 'ee_vel_y_prev','block','init_pos']
#πάρε τα id των blocks (1-104) μόνο μία φορά
blocks=df["block"].unique()
init_pos_of_each_block=[]
#Βγάλε το 0 (γιατί όταν αλλάζει το παιχνίδι από i σε i+1 ενδιάμεσα υπάρχει μια γραμμή με όλα 0)
blocks = np.delete(blocks, np.where(blocks == 0))
#Βρες τις αρχικές θέσεις κάθε block
for b in blocks:
  for i in range(len(df)):
    if df["block"].iloc[i]==b:
      init_pos_of_each_block.append(df["init_pos"].iloc[i])
      break
#οι 4 πιθανές αρχικές θέσεις
init_pos=[0,1,2,3]
#Διάλεξε 72 παιχνίδια ως background set
N=320
#Πρέπει τα παιχνίδια με ίδια αρχική θέση να είναι ισοκατανεμημένα
n_per_pos = N // len(init_pos)
sampled_indices = []

for c in init_pos:
    class_indices = np.where( init_pos_of_each_block== np.float64(c))[0]
    chosen = np.random.choice(class_indices, n_per_pos, replace=False)
    sampled_indices.extend(chosen)
sampled_indices = np.array(sampled_indices)
train_blocks = blocks[sampled_indices]
"""
"""
train_blocks=[ 59.,  95.,  33.,  75.,  91.,  66.,  45.,   5., 100.,  69.,  60.,  15.,  23.,  88.,
  50.,  30.,  28.,  73.,  12.,  71.,  85.,  74.,  20.,   3.,  47. , 81., 104.,   7.,
  57.,  21.,  61.,  68.,  92.,  49.,  43.,  35.,  58.,  41.,  32.,  17.,  24.,  13.,
  84. , 48. ,  8. , 76. ,  1.,  36.,  79.,  51.,  34. , 31.,  63. , 94.,  62.,  16.,
  82.,  53.,  39.  ,18.,  26.,  89. , 87.,  67.,  52.,  22.,  64. ,  6.,  25.,  38.,
   9.,  77.]
   """
"""
for i in range(len(df)):
    #αν η γραμμή στο df έχει block id που είναι στο train βάλε το στο training set
    if df["block"].iloc[i] in train_blocks:
        X_train_df.append(df[state_f].iloc[i])
    #αλλιώς αν δεν είναι 0 στο test set (αν είναι 0 είναι αλλαγή μεταξύ παιχιδιών άρα άχρηστο)
    else:
      if df["block"].iloc[i]!=0:
        X_test_df.append(df[state_f].iloc[i])



#-------------------------------------------------------------------------------------------------------

#κάνε τις λίστες πίνακα
X_test_all=np.array(X_test_df) #all info that i need for graphs
X_train_all=np.array(X_train_df)
#take only features for shap (4 feature states)
X_test=X_test_all[:,:4]
X_train=X_train_all[:,:4]

#-------------------------------------------------------------------------------------------------------

#φτιάξε και τα test blocks για να ξέρεις ποια είναι
test_blocks=[]
for i in range(1,104*5+1): #32->104*5
  if i not in train_blocks:
    test_blocks.append(i)
print(test_blocks)

#-------------------------------------------------------------------------------------------------------
"""
# @ Define a prediction function for generating actions for SHAP Explainer
def model_predict(X):
    X_tensor = th.tensor(X, dtype=th.float32)
    X_tensor=X_tensor.cuda()
    model.eval()
    with th.no_grad():
        return model(X_tensor).detach().cpu().numpy()

#-------------------------------------------------------------------------------------------------------

# Create the SHAP Kernel Explainer
explainer = shap.KernelExplainer(model_predict, X_train)
joblib.dump(explainer, "kernel_explainer_248_K7.pkl")
shap_values = explainer.shap_values(X_test)
shap_values = np.array(shap_values)
#-------------------------------------------------------------------------------------------------------
print(shap_values.shape)
#the returned shape is actions x samples x features. Do it samples x features x actions
shap_values=shap_values.transpose(1,2,0)
#store shap values into a csv file
shap_values_action_0=shap_values[:,:,0]
shap_values_action_1=shap_values[:,:,1]
shap_values_action_2=shap_values[:,:,2]
np.savetxt("shap_values_try1_action_0_248_K7.csv", shap_values_action_0, delimiter=",", fmt="%f")
np.savetxt("shap_values_try1_action_1_248_K7.csv", shap_values_action_1, delimiter=",", fmt="%f")
np.savetxt("shap_values_try1_action_2_248_K7.csv", shap_values_action_2, delimiter=",", fmt="%f")
#Πλέον όλα είναι αποθηκευμένα από πριν άρα λοογικά δε χρειάζεται αυτό πλέον
"""
np.savetxt("X_test_320_K1.csv",X_test,delimiter=",", fmt="%f")
with open("X_test_df_320_K1.pkl", "wb") as f:
    pickle.dump(X_test_df, f)
with open("test_blocks_320_K1.pkl", "wb") as f:
    pickle.dump(test_blocks, f)
"""