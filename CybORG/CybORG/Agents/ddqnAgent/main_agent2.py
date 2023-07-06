import inspect
from stable_baselines3 import PPO

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.dqnAgent.DQNAgent import DQNAgent


class MainAgent(BaseAgent):
    # agent that loads a StableBaselines3 PPO model file
    def train(self, results):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(CybORG(path, 'sim')))))
            print(self.use_meander)
            self.model = self.use_meander()
        action = self.model.predict(observation)
        print(action)
        return action

    def use_meander(self):
        self.agent = DQNAgent(chkpt_dir="../Agents/Models/model_meander/", algo='DQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Meander"

    def use_bline(self):
        self.agent = DQNAgent(chkpt_dir="../Agents/Models/model_b_line/", algo='DQNAgent', env_name='Scenario1b')
        # needed to get the pytorch checkpoint
        self.agent.load_models()
        self.agent_name = "Bline"