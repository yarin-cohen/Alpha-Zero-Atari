# AlphaZero- Atari- Breakout
This repository is a python implementation of DeepMind's AlphaZero reinforcement learning algorithm. The solution was
implemented for Openai's Atari-Breakout environment. It can be used to solve other environments, provided
you change specific sections.
* This is actually a modified implementation of AlphaGo-Zero algorithm, to fit single-player games, 
essentially implementing AlphaZero. 

# Overview:
* Using Tensorflow v2 but with v2 behavior disabled
* Using openai's gym

# Imlementation Details:
## model:
* A simple convolutional network, outputting both value and policy.
* Located under `train_and_play.py`, in the function `create_model_tensors()`.

## Snapshots Environment:
* To be able to play and plan games using a predefined model, we need to provide something more than just the environment
itself.
* We need to give the agent the ability to plan and predict outcomes in the atari environment. Since we don't have an
explicit model of the environment, the agent actually needs to act out a specific action it wants to plan, and to be able to
  return to the original state once it is done planning.
* To achieve this, we implemented a Wrapper class to the original gym environment. This class is implemented in
`snapshots_env.py` and is called `WithSnapshots`
* This class enables us to:
    * Take a snapshot of the current state
    * Act out various actions in the environment to be able to predict outcomes - simulating a model of the environment
    * return to the original state by using the snapshot.
* If you want to use this algorithm for different environments, you must implement your own `WithSnapshots` class, with
the proper core functions.

## MCTS:
* The file `mcts_nodes.py` implements a class of Monte Carlo search tree nodes
* Defining and initializing a node requires to provide a parent node. To define and initialize a root we need to provide
a snapshot of the current state, and the agent's observation as inputs.
* This class implements every core functionality that is detailed in the original paper.

## Playing and Training:
* The file `train_and_play.py` implements:
  * The model for both the value and policy functions of the agent.
  * The entire loop of playing the game and training the model until conversion - for X generations.
  * Every generation:
    * Loads the latest weight file for all models
    * Plays the game for X iterations
    * Collects and saves proper data
    * Trains on data - Experience Replay.

# How To Use:
* Open `global_params.py`. Enter paths to save relevant training data and plots.
* Change any other desired parameters.
* Run `train_and_play.py`

This implementation can work with other environments, with the proper adjustments:
* Change the environment in `train_and_play.py`
* Implement a proper Wrapper class for the environment to support taking snapshots - if needed
* If needed, update the model's layers and enter correct values for the state dimension and observation's dimension
in `global_params.py`
* Update any other desired parameter in `global_params.py`

    
