Learn to build, block by block.

## Vision

The overarching goal of this project is to create a world to explore and interact with.

*Prerequisite:* Said world.

This project is also about creating a bot that learns the necessary skills to build such a world.

The bot will start with simple action primitives, such as "drop a block at this coordinate". The bot will expand the set of primitives in its toolkit as it learns new skills, such as adding a wall of given dimensions or knowing how to build a column as a single action. Once the bot has been trained to reliably build a bridge across a gap, we want the bot to internalize the skill of building bridges to apply efficiently whenever it encounters a gap. The current mechanic focuses on construction by placing block after block. We will begin with a 2D grid and expand to 3D.

The bot must also learn to combine skills to accomplish higher level tasks, e.g integrating bridge-building and tower-building to cross uneven ground.

To make the world interesting, the bot will also need to learn notions of creativity and style. For example, instead of building the minimum bridge for a span, it may opt to build a covered bridge or a bridge with arches.

We will need to hone a neural architecture that allows the bot to incrementally pick up new skills and understanding. Just as a human reads a Wikipedia page about a new type of bridge, we will create an interface that allows us to easily expand the bot's knowledgebase, whether its learning a new style or facing a new challenge in its environment.

Throughout the process, we will also build tools to make our bot development, training and debugging more efficient.

Developing NPCs in the world itself is not a current project priority but would be a natural extension.

## Status

Today, the bot is learning to build simple bridges across a gap on a 2D grid.

## Setup

You will need the [gym-bridges](https://github.com/ldoshi/gym-bridges) repo for running the environment.

* Create a virtualenv with _Python 3.8.5_, and activate it.
  * If you're using python directly

    `python3 -m venv venv`

    `source venv/bin/activate`

  * If you're using miniconda

    `conda create -n myenv python=3.8.5`

    `conda activate myenv`

* Install dependencies `pip install -r requirements.txt`

This project currently uses black to autoformat all code. Highly recommend using black as well to prevent spurious diffs.

## Running

In `gym-bridges`, run `./reinstall.sh`. You will need to do this every time the gym-bridges environment code changes.

The primary code is in `bridger/bridge_builder.py`

### REPL

Set up env: `env = gym.make("gym_bridges.envs:Bridges-v0")`

Create an environment of 3x6 grid `env.setup(3, 6)`

Drop a brick (and print out score) `env.step(0)`

View the current env `env.render()`


## Training/debugging panel
