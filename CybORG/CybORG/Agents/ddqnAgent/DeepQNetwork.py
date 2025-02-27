# Checkout https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN
# The only changes we made were regarding the network architecture (not CNN here)
import pickle
import os
import torch as T
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # you may want to play around with this and forward()
        self.fc1 = nn.Linear(input_dims[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # you may want to play around with this
    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        actions = self.fc3(flat2)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        pickle.dump(T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))