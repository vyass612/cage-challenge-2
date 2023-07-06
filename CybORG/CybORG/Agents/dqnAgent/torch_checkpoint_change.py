import torch

checkpoint = torch.load('/Models_DQN/model_bline/Scenario1b_DQNAgent_q_eval')
checkpoint['fc3.weight'] = checkpoint['fc3.weight'][:41, :]
checkpoint['fc3.bias'] = checkpoint['fc3.bias'][:41]
torch.save(checkpoint, 'Scenario1b_DQNAgent_q_eval')