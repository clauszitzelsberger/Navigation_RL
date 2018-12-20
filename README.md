# Navigation_RL
Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Project "Navigation": Training an agent to navigate and collect bananas in a large, square world.

## The Challenge
The challenge is to train an agent to navigate and collect yellow bananas while avoiding blue bananas in a large, square world.
Therefore the environment provides a reward of +1 when the agent is collecting a yellow banana - a reward of -1 is provided when a blue banana is collected.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name name python=3.6
	source activate name
	```
	- __Windows__: 
	```bash
	conda create --name name python=3.6 
	activate name
	```

2. Clone the repository and navigate to the folder.  Then, install the dependencies in the `requirements.txt` file.
```bash
git clone https://github.com/clauszitzelsberger/Navigation_RL.git
cd Navigation_RL
pip install pip install -r requirements.txt
```

3. Download the Unity Environment
Download the environment that mathces your operation system, then place the file in the `Navigation_RL/` folder and unizip the file.
	- [__Linux__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	- [__Mac OSX__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
	- [__Windows (32-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
	- [__Windows (64-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `name` environment.  
```bash
python -m ipykernel install --user --name name --display-name "name"
```

5. Before running code in a notebook, change the kernel to match the `name` environment by using the drop-down `Kernel` menu. 
  
## Setup of repository

Apart from the `Readme.md` file this repository consists of the following files:

1. `agent.py`: Agent and ReplayBuffer class with all required functions
2. `model.py`: QNetwork class
3. `run.py`: Script which will train the agent. Can be run directly from the terminal.
4. `report.ipynb`: As an alternative to the `run.py` script this Jupyter Notebook has a step-by-step structure. Here the learning algorithm is described in detail
5. `checkpoint.pth`: Contains the weights of a successful QNetwork