Learn to build, block by block.

## Setup

You will need the [gym-bridges](https://github.com/ldoshi/gym-bridges) repo for running the environment.

* Create a virtualenv with _Python 3.8.5_, and activate it.
  * If you're using miniconda
  
    `conda create -n myenv python=3.8.5`
  
    `conda activate myenv`

* Install dependencies `pip install -r requirements.txt`

## Running

In `gym-bridges`, run `./reinstall.sh`. You will need to do this every time the gym-bridges environment code changes.

The primary code is in `bridge-builder/bridge_builder.py`

### REPL

Set up env: `env = gym.make("gym_bridges.envs:Bridges-v0")`

Create an environment of 3x6 grid `env.setup(3, 6)`

Drop a brick (and print out score) `env.step(0)`

View the current env `env.render()`


## Training/debugging panel
