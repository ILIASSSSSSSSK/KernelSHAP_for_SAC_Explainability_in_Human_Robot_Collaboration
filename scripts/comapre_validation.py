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

max_x= [-0.162,0.348]
min_x= [-0.350,0.159]
goal_pos=[-0.252, 0.245]

for i in range(5):
	shap_values_0 = np.loadtxt("shap_values_try1_action_0_248_K"+str(3+i)+".csv", delimiter=",")
	shap_values_1 = np.loadtxt("shap_values_try1_action_1_248_K"+str(3+i)+".csv", delimiter=",")
	shap_values_2 = np.loadtxt("shap_values_try1_action_2_248_K"+str(3+i)+".csv", delimiter=",")
	with open("X_test_final_248_K"+str(3+i)+".pkl", "rb") as f:
		X_test=pickle.load(f)
	shap_values = np.stack((shap_values_0, shap_values_1, shap_values_2), axis=2)
	print("fold: ",i)
	for j in range(2):
		print("feature: ",j)
		for a in range(3):
			print("action: ",a)
			x_R=[]
			x_L=[]
			u1=[]
			u2=[]
			u3=[]
			u4=[]
			u5=[]
			u6=[]
			for t in range(len(X_test)):
				x=X_test[t][0]*(max_x[0]-min_x[0])+min_x[0]
				if x<goal_pos[0]:
					x_L.append(shap_values[t][j][a])
				else:
					x_R.append(shap_values[t][j][a])
				ux=X_test[t][j+2]*(0.4)-0.2
				x=X_test[t][j]*(max_x[j]-min_x[j])+min_x[j]
				if (ux>0.05) and (ux<0.2):
					u1.append(shap_values[t][j+2][a])
				elif (ux<-0.05) and (ux>-0.2):
					u2.append(shap_values[t][j+2][a])
				elif (ux>=-0.05)and(ux<=0.05)and(x<goal_pos[j]-0.01):
					u3.append(shap_values[t][j+2][a])
				elif (ux>=-0.05)and(ux<=0.05)and(x>goal_pos[j]+0.01):
					u4.append(shap_values[t][j+2][a])
				elif (ux>0)and(x<=goal_pos[j]+0.01)and(x>=goal_pos[j]-0.01):
					u5.append(shap_values[t][j+2][a])
				elif (ux<0)and(x<=goal_pos[j]+0.01)and(x>=goal_pos[j]-0.01):
					u6.append(shap_values[t][j+2][a])
			#print(u6)
			avg_XR=np.mean(x_R)
			avg_XL=np.mean(x_L)
			avg_u1=np.mean(u1)
			avg_u2=np.mean(u2)
			avg_u3=np.mean(u3)
			avg_u4=np.mean(u4)
			avg_u5=np.mean(u5)
			avg_u6=np.mean(u6)
			print(avg_XR,avg_XL,avg_u1,avg_u2,avg_u3,avg_u4,avg_u5,avg_u6)






