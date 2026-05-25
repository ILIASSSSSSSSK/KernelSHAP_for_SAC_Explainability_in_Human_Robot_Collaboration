import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import copy

data_normalized=True
normalized=True

ee_vel_x_max= 0.2
ee_vel_y_max= 0.2
ee_vel_x_min= -0.2
ee_vel_y_min=-0.2
#select the experiment

#each time one experiment in heatmap
method_10_data_all=[#"/content/drive/MyDrive/Ilias_new_07102025/data/test_data_block_13.csv",
#"converted_file.csv",
#"converted_file2.csv",
#"/content/drive/MyDrive/Ilias_new3_14102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new4_14102025/data/test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new_16102025/data/test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new2_16102025/data/test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new3_16102025/data/test_data_block_13.csv"#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/test_data.csv",
]

method_10_rl_data_all=[#"/content/drive/MyDrive/Ilias_new_07102025/data/rl_test_data_block_13.csv",
                   #"rl_converted_file.csv",
                   #"rl_converted_file2.csv",
#"/content/drive/MyDrive/Ilias_new3_14102025/data/rl_test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new4_14102025/data/rl_test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new_16102025/data/rl_test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new2_16102025/data/rl_test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new3_16102025/data/rl_test_data_block_13.csv"#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/rl_test_data.csv",
]

def plot_heatmap_with_coverage(batch_number, filepaths, steps_filepaths, games_per_batch=8, threshold=0, max_x=-0.174, min_x=-0.356, max_y=0.343, min_y=0.162, smoothing_sigma=0.3, ax=None):
    all_x_coords = []
    all_y_coords = []

    # Helper function to get game data
    def get_game_data(game_number, test_data, rl_data):
        if game_number < 0 or game_number >= len(test_data):
            raise ValueError("Invalid game number.")
        start_index = 0 if game_number == 0 else int(np.sum(test_data[:game_number, -1])) + game_number
        num_rows = int(test_data[game_number, -1])


        game_data = rl_data[start_index:start_index+num_rows, :]
        return game_data

    # Helper function to get batch data
    def get_batch_data(batch_number, test_data, rl_data, games_per_batch):
        start_game = batch_number * games_per_batch

        end_game = start_game + games_per_batch
        x_coords = []
        y_coords = []
        for game_num in range(start_game, end_game):
            game_data = get_game_data(game_num, test_data, rl_data)
            x_coords.extend(game_data[:, 6])
            y_coords.extend(game_data[:, 7])
        return x_coords, y_coords

    # Iterate over each participant
    for test_data, rl_data in zip(filepaths, steps_filepaths):
        x_coords, y_coords = get_batch_data(batch_number, test_data, rl_data, games_per_batch)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
    # Create a 2D histogram for the heatmap
    print(all_x_coords)
    print(all_y_coords)
    max_x=-0.15
    min_x=-0.356
    max_y=0.36
    min_y=0.15
    heatmap, xedges, yedges = np.histogram2d(all_x_coords, all_y_coords, bins=[np.linspace(min_x, max_x, 40), np.linspace(min_y, max_y, 40)])
    total_bins = np.prod(heatmap.shape)
    filled_bins = np.nansum(heatmap > 0)
    coverage = filled_bins / total_bins
    print(heatmap)
    heatmap_normalized = heatmap #/ np.max(heatmap)

    # Apply Gaussian smoothing to the heatmap
    #smoothed_heatmap = gaussian_filter(heatmap, smoothing_sigma)
    smoothed_heatmap = gaussian_filter(heatmap_normalized, smoothing_sigma)

    # Plotting the smoothed heatmap
    if ax is None:
       	plt.figure(figsize=(10, 12), facecolor='white')

    # Mask squares with 0 counts
    masked_heatmap = np.ma.masked_where(smoothed_heatmap == 0, smoothed_heatmap)

    # Make a copy of the colormap and set masked color
    cmap = copy.copy(plt.cm.YlGn)
    cmap.set_bad(color='purple')   # color for 0-count squares

    im = ax.imshow(
        masked_heatmap.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        cmap=cmap,
        aspect='auto'
    )
    zero_patch = mpatches.Patch(color='purple', label='0 counts')
    ax.legend(handles=[zero_patch], loc='upper right',bbox_to_anchor=(1.4, 1), fontsize=11, frameon=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=18)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts', fontsize=15)
    cbar.ax.tick_params(labelsize=15)



    print(f"Coverage for Batch {batch_number}: {coverage:.2%}")

    return coverage  # Optionally return the coverage value

expert_test_data10=[]
expert_steps_test_data10=[]

batches_to_collect = [0, 3, 6]

for i in range(len(method_10_data_all)):
  expert_test_data10=[]
  expert_steps_test_data10=[]
  for expert_test_data_file_10, expert_steps_test_data_file_10 in zip([method_10_data_all[i]], [method_10_rl_data_all[i]]):
    # Load the CSV robustly
    a = np.genfromtxt(expert_steps_test_data_file_10, delimiter=',', skip_header=1)

    # If a is 1D, reshape it to 2D with one row
    if a.ndim == 1:
        a = a.reshape(1, -1)

    # Or, if it's an array of strings, convert safely
    if a.dtype.type is np.str_:
        a = np.genfromtxt(expert_steps_test_data_file_10, delimiter=',', skip_header=1, dtype=float)

    expert_test_data10.append(
        np.genfromtxt(expert_test_data_file_10, delimiter=',', skip_header=1)
    )
    expert_steps_test_data10.append(a)


  col_labels = ["Baseline", "Block 3", "Block 6"]
  row_labels = ["Normalized 75K, 4s, per 8, NN's action"]
  num_rows=len(row_labels)
  num_cols=len(col_labels)
  fig = plt.figure(figsize=(30, 8), dpi=50)
  for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # TL Participant
            #print(expert_steps_test_data10)
            plot_heatmap_with_coverage(batch_number, expert_test_data10, expert_steps_test_data10, ax=ax)
            # Plot as separate figure
            fig_single, ax_single = plt.subplots(figsize=(8, 6))

            plot_heatmap_with_coverage(
                batch_number,
                expert_test_data10,
                expert_steps_test_data10,
                ax=ax_single
            )

            ax_single.set_title(title, fontsize=20)

        ax.set_title(title, fontsize=20)


        ax.set_title(title,fontsize=20)

  # Adjust the layout
  plt.tight_layout()

  # Show the figure
  plt.show()


#-------------------------------------------------------------------------------------------------------
average_wins=[]
average_norm_dist=[]
average_rewards=[]

def plot_wins(method,fig,axs,color1="red",color2="red",name=""):
	count=0
	for i in method:
		df=pd.read_csv(i)
		#print(df)
		reward=df["Rewards"]+150
		reward_wins=df["Rewards"]
		norm_dist=df["Travelled Distance"]*df["Episodes Duration in Seconds"]/30

		wins=[]
		avg_reward=[]
		avg_norm_dist=[]
		for j in range(0,len(reward),8):
			avg_reward.append(np.average(reward[j:(j+8)]))
			avg_norm_dist.append(np.average(norm_dist[j:(j+8)]))
			w=0
			for l in range(8):

				if reward_wins[j+l]>-150:
					w+=1
			wins.append(w)
		print(avg_reward)
		print(wins)
		print(avg_norm_dist)
		if count==0:
			if name=="":
				axs[0].plot(range(0,7),avg_reward, 'o-',color=color1)
			else:
				axs[0].plot(range(0,7),avg_reward, 'o-',color=color1)
		else:
			axs[0].plot(range(0,7),avg_reward, 'o-',color=color1)
		axs[0].set_title("Average score per Block")
		axs[0].set(xlabel='Block', ylabel='score')
		axs[0].set_ylim(0,165)
		#plt.title('Averge reward per Block')


		if count==0:
			if name=="":
				axs[1].plot(range(0,7),wins, 'o-',color=color2)
			else:
				axs[1].plot(range(0,7),wins, 'o-',color=color2)
		else:
			axs[1].plot(range(0,7),wins, 'o-',color=color2)
		axs[1].set_ylim(0,9)
		axs[1].set_title("Wins per Block")
		axs[1].set(xlabel='Block', ylabel='Wins')

		if count==0:
			if name=="":
				axs[2].plot(range(0,7),avg_norm_dist, 'o-',color=color2)
			else:
				axs[2].plot(range(0,7),avg_norm_dist, 'o-',color=color2)
		else:
			axs[2].plot(range(0,7),avg_norm_dist, 'o-',color=color2)
		axs[2].set_ylim(0,1.3)
		axs[2].set_title("Average Normalized Distance per Block")
		axs[2].set(xlabel='Block', ylabel='Normalized Distance')
		count=1
		average_wins.append(wins)
		average_norm_dist.append(avg_norm_dist)
		average_rewards.append(avg_reward)

method_10_data=[#"/content/drive/MyDrive/Ilias_new2_16102025/data/test_data_block_13.csv",
#"/content/drive/MyDrive/Ilias_new2_14102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new_07102025/data/test_data_block_13.csv",
#"converted_file.csv",
#"converted_file2.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new4_14102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new3_14102025/data/test_data_block_13.csv"#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/test_data.csv",
]
fig,axs=plt.subplots(1,3, figsize=(17, 4),dpi=100)

plot_wins(method_10_data,fig=fig,axs=axs,color1="red",color2="red",name="")
# Create separate figures from each subplot
for ax in axs:

    fig_new, ax_new = plt.subplots(figsize=(6, 4))

    # Copy lines
    for line in ax.get_lines():
        ax_new.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            label=line.get_label()
        )

    # Copy titles/labels
    ax_new.set_title(ax.get_title())
    ax_new.set_xlabel(ax.get_xlabel())
    ax_new.set_ylabel(ax.get_ylabel())

    # Copy legend if exists
    if ax.get_legend():
        ax_new.legend()

plt.show()