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

shap_values_0 = np.loadtxt("shap_values_try1_action_0_248_K3.csv", delimiter=",")
shap_values_1 = np.loadtxt("shap_values_try1_action_1_248_K3.csv", delimiter=",")
shap_values_2 = np.loadtxt("shap_values_try1_action_2_248_K3.csv", delimiter=",")
with open("X_test_final_248_K3.pkl", "rb") as f:
    X_test=pickle.load(f)
shap_values = np.stack((shap_values_0, shap_values_1, shap_values_2), axis=2)
max_x= -0.162
min_x= -0.350
max_y= 0.348
min_y= 0.159
x_bins=np.linspace(min_x, max_x, 40)
y_bins=np.linspace(min_y, max_y, 40)

def shap_heatmaps(feature,action):
	shap_heatmap_avg_a=np.zeros((len(y_bins),len(x_bins)))
	shap_heatmap_std_a=np.zeros((len(y_bins),len(x_bins)))
	for j in range(1,len(x_bins)):
			for l in range(1,len(y_bins)):
				i_in_j_l_state=[]
				for i in range(len(X_test)):
					if (x_bins[j-1]<=(X_test[i][0]*(max_x-min_x)+min_x)<x_bins[j])and(y_bins[l-1]<=(X_test[i][1]*(max_y-min_y)+min_y)<y_bins[l]):
						i_in_j_l_state.append(shap_values[i][feature][action])
				if len(i_in_j_l_state)==0:
					#equal to -10 if the EE did not visit this state (no shap values are smaller than -0.6)
					shap_heatmap_avg_a[l][j]=-10 
					shap_heatmap_std_a[l][j]=-10
					continue
				#print(i_in_j_l_state)
				avg=np.mean(i_in_j_l_state)
				std=np.std(i_in_j_l_state)
				shap_heatmap_avg_a[l][j]=avg 
				shap_heatmap_std_a[l][j]=std 
	return shap_heatmap_avg_a,shap_heatmap_std_a

def speed_heatmaps(feature):
	speed_heatmap_avg_a=np.zeros((len(y_bins),len(x_bins)))
	speed_heatmap_std_a=np.zeros((len(y_bins),len(x_bins)))
	for j in range(1,len(x_bins)):
			for l in range(1,len(y_bins)):
				i_in_j_l_state=[]
				for i in range(len(X_test)):
					if (x_bins[j-1]<=(X_test[i][0]*(max_x-min_x)+min_x)<x_bins[j])and(y_bins[l-1]<=(X_test[i][1]*(max_y-min_y)+min_y)<y_bins[l]):
						i_in_j_l_state.append((X_test[i][feature]*0.4-0.2))
				if len(i_in_j_l_state)==0:
					#equal to -10 if the EE did not visit this state (no shap values are smaller than -0.6)
					speed_heatmap_avg_a[l][j]=-10 
					speed_heatmap_std_a[l][j]=-10
					continue
				#print(i_in_j_l_state)
				avg=np.mean(i_in_j_l_state)
				std=np.std(i_in_j_l_state)
				speed_heatmap_avg_a[l][j]=avg 
				speed_heatmap_std_a[l][j]=std 
	return speed_heatmap_avg_a,speed_heatmap_std_a

def plot_heatmaps(feature,title,title_speed=""):
 shap_heatmap_a0,std_shap_heatmap_a0=shap_heatmaps(feature,0)
 shap_heatmap_a1,std_shap_heatmap_a1=shap_heatmaps(feature,1)
 shap_heatmap_a2,std_shap_heatmap_a2=shap_heatmaps(feature,2)
 speed_heatmap=[]
 std_speed_heatmap=[]
 if (feature==2) or (feature==3):
 	speed_heatmap,std_speed_heatmap=speed_heatmaps(feature)
 
 heatmaps=[shap_heatmap_a0,shap_heatmap_a1,shap_heatmap_a2]

 if (feature==2) or (feature==3):
 	heatmaps=[speed_heatmap,shap_heatmap_a0,shap_heatmap_a1,shap_heatmap_a2]
 
 fig, axes = plt.subplots(1,3)
 actions=["SHAP in Action 0","SHAP in Action 1","SHAP in Action -1"]
 if len(heatmaps)==4:
 	fig,axes=plt.subplots(1,4)
 	plt.subplots_adjust(wspace=0.4)
 	actions=[title_speed,"SHAP in Action 0","SHAP in Action 1","SHAP in Action -1"]
 i=0
 for ax in axes:

	 masked_shap_heatmap_a1 = np.ma.masked_where(heatmaps[i] == -10, heatmaps[i])
	 cmap_name="coolwarm"
	 if (feature==2) or (feature==3):
	 	if i==0:
	 		cmap_name="rainbow"
	 cmap = copy.copy(mpl.cm.get_cmap(cmap_name))
	 cmap.set_bad('black')
	 vmin=-0.64
	 vmax=0.64
	 if (feature>1)and(i==0):
	 	vmin=-0.2
	 	vmax=0.2
	 im=ax.imshow(masked_shap_heatmap_a1,extent=[min_x, max_x, min_y, max_y], origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
	 zero_patch = mpatches.Patch(color='black', label='0 counts')
	 if ((i==2)and(feature<2))or((i==3)and(feature>1)):
	 	x_coord=1.7
	 	if feature>1:
	 		x_coord=2
	 	ax.legend(handles=[zero_patch], loc='upper right',bbox_to_anchor=(x_coord, 1), fontsize=11, frameon=True)
	 ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	 ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	 ax.tick_params(axis='both', labelsize=12)
	 cbar = plt.colorbar(im, ax=ax,shrink=0.5)
	 label_title='Average SHAP value in the discrete state (x,y)'
	 if feature==2:
	 	label_title='Average u_x value in the discrete state (x,y)'
	 elif feature==3:
	 	label_title='Average u_y value in the discrete state (x,y)'
	 cbar.set_label(label_title, fontsize=11)
	 cbar.ax.tick_params(labelsize=10)
	 ax.set_title(actions[i])
	 ax.set_xlabel("X (m)")
	 ax.set_ylabel("Y (m)")
	 goal_pos=[-0.252, 0.245]
	 circle = patches.Circle((goal_pos[0], goal_pos[1]),radius=0.01,edgecolor='purple',facecolor='none',linewidth=2)
	 ax.add_patch(circle)
	 ax.scatter([goal_pos[0]],[goal_pos[1]],color='purple',s=50,zorder=5)
	 i+=1
 fig.suptitle(title,y=0.8)
 plt.show()
titles=["SHAP value of x in each action for each discrete state (x,y)","SHAP value of y in each action for each discrete state (x,y)",
"SHAP value of u_x in each action for each discrete state (x,y)","SHAP value of u_y in each action for each discrete state (x,y)"]
plot_heatmaps(0,titles[0])
plot_heatmaps(1,titles[1])
plot_heatmaps(2,titles[2],"Avg u_x")
plot_heatmaps(3,titles[3],"Avg u_y")


