from torch import tensor, float, load
from torch.nn import Module, ModuleList, Linear, BatchNorm1d
from torch.nn.functional import relu
from typing import List


class Actor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = None

    def curr_pos(self):
        return self.x, self.y


class Player(Actor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = 'player'


class Enemy(Actor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = 'enemy'


class DQN(Module):

    def __init__(self, inputs=25, hidden_depth=1, hidden_dim=16, outputs=5):
        super(DQN, self).__init__()
        self._input = Linear(inputs, hidden_dim)

        self._hidden_B = ModuleList([BatchNorm1d(hidden_dim) for _ in range(hidden_depth)])
        self._hidden_L = ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth)])
        self._output = Linear(hidden_dim, outputs)

    def forward(self, x):
        global layers

        x = relu(self._input(x.view(x.size(0), -1)))
        layers[0] = x
        # noinspection PyTypeChecker
        for batch_norm, linear in zip(self._hidden_B, self._hidden_L):
            x = relu(batch_norm(linear(x)))
        layers[1] = x
        return self._output(x)


def state(actors_in, n=5, m=5):
    state_res = [[10] * n for _ in range(m)]
    for actor in actors_in:
        x, y = actor.curr_pos()
        state_res[y][x] = 127 if actor.type is 'enemy' else 255
    return tensor(state_res, dtype=float).unsqueeze(0)


if __name__ == '__main__':
    layers = [None for _ in range(3)]
    NET_PATH = './mem/policy_net.pkl'
    actors_list = [[Player(2, 2), Enemy(4, 0), Enemy(0, 4)],
                   [Player(0, 0), Enemy(4, 0), Enemy(0, 4)],
                   [Player(0, 0), Enemy(1, 1), Enemy(0, 4)],
                   [Player(0, 0), Enemy(0, 4), Enemy(1, 1)]]

    net = DQN()
    net.load_state_dict(load(NET_PATH))
    net.eval()

    s = []
    for actors in actors_list:
        s.append(state(actors))

    layers[2] = net(s[1])
