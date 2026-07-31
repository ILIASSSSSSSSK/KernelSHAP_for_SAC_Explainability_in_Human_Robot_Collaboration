A GUIDE FOR THE USE OF THE SIMULATION
---------------------------------------
The specific readme includes directions to run the gazebo simulation of a virtual collaborative task used in the thesis "Explainability in Human-Artificial Intelligence Collaboration" written by Ilias Kasiotakis, submitted in partial fulfilment of the requirements for the degree of Master of Artificial Intelligence at the UNIVERSITY OF PIRAEUS. In the virtual collaborative task, a human and a Deep Reinforcement Learning (DRL) agent jointly control the End-Effector (EE) of a UR3 cobot toward a target position at a controlled speed. The human controls the motion of the EE in the y-axis and the DRL agent controls the motion of the EE in the x-axis. The EE can start from 4 possible initial positions that form a rectangular parallelogram. To represent the four possible initial positions, four blue cylindrical markers are used, centered on the coordinates that correspond to those positions. To represent the target position, a red cylinder is used with its center at the coordinates of the target position. Finally, the current position of the EE is represented by a green cylindrical marker, whose center is shifted based on the coordinates of the current EE’sposition. The Gazebo simulation of the collaborative task is presented in the following figure:

![image alt](https://github.com/ILIASSSSSSSSK/KernelSHAP_for_SAC_Explainability_in_Human_Robot_Collaboration/blob/99f49fc1ab73c47b747709de56d2272307073d0d/gazebo_simulation.jpg)

---------------
WHAT IS NEEDED
---------------

1. To launch the game with the simulation run:
```bash
   source ../../opt/ros/melodic/setup.bash
   source Desktop/catkin_ws5/devel/setup.bash 
   roslaunch human_robot_collaborative_learning game_with_gazebo.launch 
```
2. In order to rephresh the EE's green cylindrical marker position at each timestep run:
```bash
   source ../../opt/ros/melodic/setup.bash
   source Desktop/catkin_ws5/devel/setup.bash 
   source Desktop/spawn_marker_catkin_ws/devel/setup.bash 
   rosrun spawn_maker_pkg listener.py 
```
The code of the spawn_marker_catkin_ws can be found in the following github repository: https://github.com/ILIASSSSSSSSK/spawn_marker_catkin_ws

3. To control the EE with the keyboard run:
```bash
   source ../../opt/ros/melodic/setup.bash
   rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
```
**Important note**: The terminal where the teleop_twist_keyboard command was executed must be right-clicked to allow the user to control the motion of the EE along the y-axis. The EE motion is controlled using the following keys: 
- ‘i’: positive acceleration.
- ‘,’: negative acceleration.
- ‘k’: zero acceleration.

---------------------------------------------------
Important parameters (in rl_params.yaml in config)
---------------------------------------------------


- train_model: Set to True if you want the entire game to be executed (including the training and testing blocks). Set to False if you only want to test an existing trained agent.

- num_blocks: If train_model=False, set num_blocks equal to 1  

- load_model_testing_dir_actor and load_model_testing_dir_critic: The paths to the actor and critic network files used when testing an agent.

- initialized_agent: True if you want a specific initialized agent in the baseline block (this option is usefull only if train_model=True). Otherwise, set it to False.

- initialized_agent_dir: The directory where the initialized actor and critic network files are stored.

- gazebo_simulation: Set to True if you want to run the game using the Gazebo simulation.
  
- participant_name: Make sure to change this value when running a new game.


--------------------------------------------------------------
Important parameters (in robot_control_params.yaml in config)
--------------------------------------------------------------


- max_vel: maximum value of EE's velocity (same for both x and y axis)
- min_vel: minimum value of EE's velocity (same for both x and y axis)

- max_x: maximum value of the x position of the EE
- min_x: minimum value of the x position of the EE
- max_y: maximum value of the y position of the EE
- min_y: minimum value of the y position of the EE


---------------
Important Notes
---------------
1. The initial positions are defined in the game_control_sign.py file as:
```bash
   position0_config=[ -0.9691837469684046, -2.057300869618551, -1.3772237936602991, -2.749833885823385, -2.679509703313009,  -18.8495]
   position1_config=[ -0.6513956228839319, -2.423556152974264,-0.7467053572284144, -3.0507639090167444, -2.3643069903003138, -18.8495]
   position2_config=[-0.48472386995424444, -1.411879841481344, -2.149219814931051, -2.6642029921161097, -2.1951726118670862, -18.8495]
   position3_config=[-0.22891742387880498, -1.9096925894366663, -1.5949791113482874,-2.7291744391070765, -1.9404695669757288,-18.8495]
```
based on the configuration of the real robot on those positions. If new initial positions are needed the new corresponding configurations need to be obtained and the values of those lists need to be changed.

2. The camera viewpoint of the Gazebo simulation must be adjusted manually.
   
3. Details about the installation of the collaborative game can be found here: https://github.com/Roboskel-Manipulation/hrc_study_tsitosetal. After the installation is completed, replace the contents of hrc_study_tsitosetal folder with the contents of this repository.

4. After the installation is completed, replace the file ur3_table_v2020.urdf.xacro located in /manos_description/urdf/ with the file provided here: https://github.com/ILIASSSSSSSSK/manos_new-edition/blob/main/manos_description/urdf/ur3_table_v2020.urdf.xacro. This replacement is necessary to make the markers indicating the initial possible positions and the target position visible.

5. In https://github.com/ILIASSSSSSSSK/KernelSHAP_for_SAC_Explainability_in_Human_Robot_Collaboration/tree/main/scripts, a README file is included that provides instructions about the files used for the implementation of the game and the KernelSHAP Explainer.
