# Navigation_RL
Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Project "Navigation": Training an agent to navigate and collect bananas in a large, square world.

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
  
