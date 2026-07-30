A GUIDE FOR THE USE OF THE SIMULATION
---------------------------------------


---------------
WHAT IS NEEDED
---------------

1. To launch the game with the simulation run:
```bash
   source ../../opt/ros/melodic/setup.bash
   source Desktop/catkin_ws5/devel/setup.bash 
   roslaunch human_robot_collaborative_learning game_with_gazebo.launch 
```
2. In order to rephresh the EE marker position at each timestep run:
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
Important note: the

---------------------------------------------------
Important parameters (in rl_params.yaml in config)
---------------------------------------------------


- train_model: True if you want the whole game to be run (training and testing blocks). IF you want only to TEST an exisiting trained agent use False

- num_blocks: If train_model=False, then set num_blocks to be equal to 1 (error otherwise if I remember correctly) 

- load_model_testing_dir_actor and load_model_testing_dir_critic: the locations of the actor and the critic when you want to TEST an agent

- initialized_agent: True if you want a specific initialized agent in the agent block (this is usefull only if train_model=True), False otherwise

- initialized_agent_dir: Where the initialized agent actor and critic are stored

- gazebo_simulation: True if you want to run the game in the gazebo simulation

- participant_name: Be carefull to change it when you run a new game


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
based on the configuration of the real robot on those positions. If new initial positions are needed the new corresponding configurations need to be obtained and the values of those lists need to be changed

2. The simulation starts 
