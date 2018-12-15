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

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `name` environment.  
```bash
python -m ipykernel install --user --name name --display-name "name"
```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]
  
