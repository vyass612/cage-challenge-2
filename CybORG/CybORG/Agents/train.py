import inspect
import time
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.dqnAgent import dqn_agent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

def wrap( env):
    return ChallengeWrapper('Blue', env)

# def wrap(env):
#    return OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))


if __name__ == "__main__":
    #The cybORG stuff
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    agent_name = 'Blue'

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    red_agents = [B_lineAgent]

    RL_algos = ["DQN"]

    timesteps = 100000

    steps = round(timesteps/1000000, 2)

    for red_agent in red_agents:
        for RL_algo in RL_algos:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})

            env = wrap(cyborg)

            model = RLAgent(env=env, agent_type = RL_algo)

            model.train(timesteps=int(timesteps), log_name = f"{RL_algo}")

            model.save(f"{RL_algo} against {red_agent.__name__}")
        
