import shap
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

shap_values_0 = np.loadtxt("shap_values_try1_action_0_248_K4.csv", delimiter=",")
shap_values_1 = np.loadtxt("shap_values_try1_action_1_248_K4.csv", delimiter=",")
shap_values_2 = np.loadtxt("shap_values_try1_action_2_248_K4.csv", delimiter=",")
with open("X_test_final_248_K4.pkl", "rb") as f:
    X_test=pickle.load(f)
shap_values = np.stack((shap_values_0, shap_values_1, shap_values_2), axis=2)
max_x= -0.162
min_x= -0.350
max_y= 0.348
min_y= 0.159
goal_pos=[-0.252, 0.245]
pos_q1=[]
pos_q2=[]
u_in_pos_q1=[]
u_shap_a0_q1=[]
u_shap_a1_q1=[]
u_shap_a2_q1=[]

u_in_pos_q2=[]
u_shap_a0_q2=[]
u_shap_a1_q2=[]
u_shap_a2_q2=[]
#0 for x, 1 for y
pos_sel=0
u_name='u_y'
pos_name='y'
q1_name=' when x>x_target'
q2_name=' when x<=x_target'
for i in range(len(X_test)):
    multi=max_x-min_x
    add=min_x
    q_lim=goal_pos[1]
    if  pos_sel==1:
        multi=max_y-min_y
        add=min_y
        q_lim=goal_pos[0]
    pos=X_test[i][1]*(max_y-min_y)+min_y
    if pos_sel==1:
        pos=X_test[i][0]*(max_x-min_x)+min_x
    if pos>q_lim:
        pos_q1.append(X_test[i][pos_sel]*multi+add)
        u_in_pos_q1.append(X_test[i][pos_sel+2]*0.4-0.2)
        u_shap_a0_q1.append(shap_values[i][pos_sel+2][0])
        u_shap_a1_q1.append(shap_values[i][pos_sel+2][1])
        u_shap_a2_q1.append(shap_values[i][pos_sel+2][2])
    else:
        pos_q2.append(X_test[i][pos_sel]*multi+add)
        u_in_pos_q2.append(X_test[i][pos_sel+2]*0.4-0.2)
        u_shap_a0_q2.append(shap_values[i][pos_sel+2][0])
        u_shap_a1_q2.append(shap_values[i][pos_sel+2][1])
        u_shap_a2_q2.append(shap_values[i][pos_sel+2][2])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
axes_flat = axes.flatten()
sc1=axes_flat[0].scatter(pos_q1, u_in_pos_q1,c=u_shap_a0_q1, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[0].set_title("Action 0")

sc2=axes_flat[1].scatter(pos_q1, u_in_pos_q1,c=u_shap_a1_q1, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[1].set_title("Action 1")

sc3=axes_flat[2].scatter(pos_q1, u_in_pos_q1,c=u_shap_a2_q1, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[2].set_title("Action -1")
fig.colorbar(sc1, ax=axes_flat[2], label='SHAP value of '+u_name+q1_name)
for i in range(3):
    axes_flat[i].axvline(x=goal_pos[pos_sel]+0.01, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axvline(x=goal_pos[pos_sel]-0.01, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axhline(y=0.05, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axhline(y=-0.05, linestyle='--', color='black', linewidth=1)
    axes_flat[i].set_xlabel(pos_name+"(m)")
    axes_flat[i].set_ylabel(u_name+"(m/s)")
    minn=min_x
    maxx=max_x
    if pos_sel==1:
        minn=min_y
        maxx=max_y
    axes_flat[i].set_xlim(minn, maxx)
    axes_flat[i].set_ylim(-0.2, 0.2)
fig.suptitle("SHAP value of "+u_name+" according to the value of "+u_name+" and "+pos_name+" of the EE")

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
axes_flat = axes.flatten()
sc4=axes_flat[0].scatter(pos_q2, u_in_pos_q2,c=u_shap_a0_q2, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[0].set_title("Action 0")

sc5=axes_flat[1].scatter(pos_q2, u_in_pos_q2,c=u_shap_a1_q2, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[1].set_title("Action 1")

sc6=axes_flat[2].scatter(pos_q2, u_in_pos_q2,c=u_shap_a2_q2, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes_flat[2].set_title("Action -1")


fig.colorbar(sc6, ax=axes_flat[2], label='SHAP value of '+u_name+q2_name)
for i in range(3):
    axes_flat[i].axvline(x=goal_pos[pos_sel]+0.01, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axvline(x=goal_pos[pos_sel]-0.01, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axhline(y=0.05, linestyle='--', color='black', linewidth=1)
    axes_flat[i].axhline(y=-0.05, linestyle='--', color='black', linewidth=1)
    axes_flat[i].set_xlabel(pos_name+"(m)")
    axes_flat[i].set_ylabel(u_name+"(m/s)")
    minn=min_x
    maxx=max_x
    if pos_sel==1:
        minn=min_y
        maxx=max_y
    axes_flat[i].set_xlim(minn, maxx)
    axes_flat[i].set_ylim(-0.2, 0.2)
fig.suptitle("SHAP value of "+u_name+" according to the value of "+u_name+" and "+pos_name+" of the EE")

plt.tight_layout()
plt.show()


pos_x=[]
pos_y=[]

pos_shap_a0=[]
pos_shap_a1=[]
pos_shap_a2=[]

pos_sel=0

for i in range(len(X_test)):
    pos_y.append(X_test[i][1]*(max_y-min_y)+min_y)
    pos_x.append(X_test[i][0]*(max_x-min_x)+min_x)
    pos_shap_a0.append(shap_values[i][pos_sel][0])
    pos_shap_a1.append(shap_values[i][pos_sel][1])
    pos_shap_a2.append(shap_values[i][pos_sel][2])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sc1=axes[0].scatter(pos_x, pos_y,c=pos_shap_a0, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes[0].set_title("Action 0")

sc1=axes[1].scatter(pos_x, pos_y,c=pos_shap_a1, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes[1].set_title("Action 1")

sc1=axes[2].scatter(pos_x, pos_y,c=pos_shap_a2, cmap='rainbow', vmin=-0.64, vmax=0.64)
axes[2].set_title("Action -1")

fig.colorbar(sc1, ax=axes[2], label='SHAP value of '+pos_name)

for i in range(3):
    
    axes[i].set_xlabel("X(m)")
    axes[i].set_ylabel("Y(m)")
    axes[i].set_xlim(min_x, max_x)
    axes[i].set_ylim(min_y, max_y)
    circle = patches.Circle((goal_pos[0], goal_pos[1]),radius=0.01,edgecolor='purple',facecolor='none',linewidth=2)
    axes[i].add_patch(circle)
    axes[i].scatter([goal_pos[0]],[goal_pos[1]],color='purple',s=50,zorder=5)
fig.suptitle("SHAP value of "+pos_name+" according to the value of x and y of the EE")

plt.tight_layout()
plt.show()