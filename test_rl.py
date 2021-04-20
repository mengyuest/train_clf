import torch
from torch import nn
import gym
import numpy as np

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        network = []
        network.append(nn.Linear(3, 256))
        network.append(nn.ReLU())
        network.append(nn.Linear(256, 128))
        network.append(nn.ReLU())
        network.append(nn.Linear(128, 64))
        network.append(nn.ReLU())
        network.append(nn.Linear(64, 1))
        network.append(nn.Tanh())

        self.net = nn.Sequential(*network)

    def load(self, path="../DeepReinforcementLearning/models/1450.ckpt"):
        m = torch.load(path)

        state_dict=self.state_dict()
        for key in state_dict:
            state_dict[key] = m[key.split("net.")[1]]
        self.load_state_dict(state_dict)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    seed=1007
    torch.manual_seed(seed)
    env=gym.make("Pendulum-v0")
    env.seed(seed)
    num_steps=300

    actor = Actor()
    actor.load()

    s = env.reset()

    th_list=[]
    thdot_list=[]
    for t in range(num_steps):
        th_list.append(np.arctan2(s[1],s[0]))
        thdot_list.append(s[2])

        s = torch.from_numpy(s).float().unsqueeze(0)
        u = actor(s)
        u = u.squeeze(0).detach().cpu()
        s, reward, _, _ = env.step(u)
        env.render()

    import matplotlib.pyplot as plt
    # plt.plot(th_list, thdot_list)
    plt.plot([np.sin(x) for x in th_list],thdot_list)
    plt.show()