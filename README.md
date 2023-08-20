# DRND_Continuous_Control
## Project Description
For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Unity ML-Agents Reacher Environment](https://media.giphy.com/media/doeuwYIlGFE50GS6Hq/giphy.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, we will use a single agent environment.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Installation
The project was in a linux server with unityagent=0.4.0 and python 3.6 installed.

1. You may need to install Anaconda and create a python 3.6 environment.
```bash
conda create -n drnd python=3.6
conda activate drnd
```
2. Clone the repository below, navigate to the python folder and install dependencies. Pay attention that the torch=0.4.0 in the requirements.txt is no longer listed in pypi.org, you may leave your current torch and remove the line torch=0.4.0
 ```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
3. Download unity environment file  [reacher](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip). This is the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.
(To watch the agent, you may follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the [environment for the Linux operating system](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip).)

5. Unzip the compressed file
6. Create the Ipython kernel:
```bash
python -m ipykernel install --user --name=drnd
```

   
## Executing 
In the notebook, be sure to change the kernel to match "drnd" by using the drop down in "Kernel" menu. Be sure the adjust the Reacher file location locally.

Executing Continuous_Control.ipynb
  
