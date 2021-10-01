Learn to build, block by block.

## Vision

The overarching goal of this project is to create a world to explore and interact with.

*Prerequisite:* Said world.

This project is also about creating a bot that learns the necessary skills to build such a world.

The bot will start with simple action primitives, such as "drop a block at this coordinate". The bot will expand the set of primitives in its toolkit as it learns new skills, such as adding a wall of given dimensions or knowing how to build a column as a single action. Once the bot has been trained to reliably build a bridge across a gap, we want the bot to internalize the skill of building bridges to apply efficiently whenever it encounters a gap. The current mechanic focuses on construction by placing block after block. We will begin with a 2D grid and expand to 3D.

The bot must also learn to combine skills to accomplish higher level tasks, e.g integrating bridge-building and tower-building to cross uneven ground.

To make the world interesting, the bot will also need to learn notions of creativity and style. For example, instead of building the minimum bridge for a span, it may opt to build a covered bridge or a bridge with arches.

We will need to hone a neural architecture that allows the bot to incrementally pick up new skills and understanding. Just as a human reads a Wikipedia page about a new type of bridges, we will create an interface that allows us to easily expand the bot's knowledgebase, whether its learning a new style or facing a new challenge in its environment.

Throughout the process, we will also build tools to make our bot development, training and debugging more efficient.

Developing NPCs in the world itself is not a current project priority but would be a natural extension.

## Status

Today, the bot is learning to build simple bridges across a gap on a 2D grid.

## Setup

You will need the [gym-bridges](https://github.com/ldoshi/gym-bridges) repo for running the environment.

* Create a virtualenv with _Python 3.9.6_, and activate it.
  * If you're using python directly

    `python3 -m venv venv`

    `source venv/bin/activate`

  * If you're using miniconda

    `conda create -n rome python=3.9.6`

    `conda activate rome`

* Install dependencies (tested for conda)

1. Navigate to the gym-bridges environment and run `./reinstall.sh`. This installs `gym-bridges` within the environment. If you don't do this, running any installation in `rome-wasnt-built-in-a-day` will not be able to install from `setup.py`.
2. Run the command `pip3 install -e .`. This installs the current directory as a package using the information from `setup.py`. 
3. Run the command `pip3 install -r requirements.txt`. This installs the packages required to run the library.

This project currently uses black to autoformat all code. Highly recommend using black as well to prevent spurious diffs.

## Running

In `gym-bridges`, run `./reinstall.sh`. You will need to do this every time the gym-bridges environment code changes.

Then, from this repository, call `pip install .` to install this package (`bridger`).

The main testing entrypoint is to call `bridge_builder.py` from command line. Default arguments are available for all inputs, and can be viewed by calling `bridge_builder.py --help`

## Debugging

If the `debug` flag is set, training history details will be stored to the directory indicated by flag `training_history_dir`. 

Run `training_viewer.py` to launch the debugging visualization tool reading data from the directory indicated by flag `training_history_dir`. 
