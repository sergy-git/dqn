from q_tools import *
from random import choice, random, sample
from time import sleep
from typing import List
from torch import tensor, float, stack
from torch.nn import Module, Conv2d, BatchNorm2d, Linear
from torch.optim import RMSprop as Optimizer
from torch.nn import SmoothL1Loss as Loss
from torch.nn.functional import relu
from collections import namedtuple


Transition = namedtuple('Transition', ('prev_state', 'last_action', 'curr_state', 'reward'))


class Actor:
    def __init__(self, x, y):
        self.init_x = x
        self.init_y = y
        self.x = self.init_x
        self.y = self.init_y
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.type = None

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.x_prev = None
        self.y_prev = None
        self.action = None

    def curr_pos(self):
        return self.x, self.y

    def prev_pos(self):
        return self.x_prev, self.y_prev

    def next_pos(self, action):
        return self.x + action[0], self.y + action[1]

    def move(self, policy_strategy, valid_pos):
        action = policy_strategy()
        if not valid_pos(self.next_pos(action)):
            action = (0, 0)
        x_new, y_new = self.next_pos(action)
        self.action = action
        self.x_prev = self.x
        self.y_prev = self.y
        self.x = x_new
        self.y = y_new


class Player(Actor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = 'player'


class Enemy(Actor):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = 'enemy'


class SimpleBoard:
    def __init__(self, n, m, actors):
        self.n = n
        self.m = m
        self.actors = actors

    def draw(self):
        rows = [['[ ]'] * self.n for _ in range(self.m)]
        for actor in self.actors:
            x, y = actor.curr_pos()
            if actor.type is 'enemy':
                rows[y][x] = '[x]'
            else:
                rows[y][x] = '[@]'

        print('\n' * 5)  # Clear screen
        for row in rows:
            for cell in row:
                print(cell, end='')
            print()
        print('___' * self.n)


class World:
    player: Actor
    enemies: List[Actor]
    actors: List[Actor]

    def __init__(self, policy_strategy, n=5, m=5):
        self.actions = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
        self.keys = {self.actions[index]: index for index in range(len(self.actions))}
        self.policy = policy_strategy
        self.n = n
        self.m = m
        self.player = Player(0, 0)
        self.enemies = [Enemy(0, 2), Enemy(2, 2)]
        self.actors = [self.player] + self.enemies
        self.board = SimpleBoard(self.n, self.m, self.actors)
        self._step_count = 0
        self._reward = 0
        self._prev_state = None
        self._last_action = None
        self._curr_state = self.state()

    def step_count(self):
        return self._step_count

    def reward(self):
        return tensor(self._reward, dtype=float)

    def prev_state(self):
        return self._prev_state

    def last_action(self):
        return tensor(self.keys[self._last_action])

    def curr_state(self):
        return self._curr_state

    def transition(self):
        return Transition(self.prev_state(), self.last_action(), self.curr_state(), self.reward())

    def reset(self):
        self._step_count = 0
        for actor in self.actors:
            actor.reset()
        self._curr_state = self.state()

    def draw(self):
        sleep(1)
        self.board.draw()
        print('Steps:', self._step_count)

    def valid_pos(self, position):
        x, y = position
        return 0 <= x < self.n and 0 <= y < self.m

    def game_over(self):
        game_over = any(map(lambda enemy: enemy.curr_pos() == self.player.curr_pos(), self.enemies))
        if game_over:
            self._reward = -1
        else:
            self._reward = 1 if not self._last_action == (0, 0) else .9
        return game_over

    def random_action(self):
        return choice(self.actions)

    def strategy(self):
        return self.actions[self.policy(self.state())]

    def play(self, silent=False):
        self.player.move(self.strategy, self.valid_pos)
        for enemy in self.enemies:
            enemy.move(self.random_action, self.valid_pos)
        self._prev_state = self._curr_state
        self._last_action = self.player.action
        self._curr_state = self.state()
        self._step_count += 1
        if not silent:
            self.draw()
        return not self.game_over()

    def state(self):
        state = [[10] * self.n for _ in range(self.m)]
        for actor in self.actors:
            x, y = actor.curr_pos()
            state[x][y] = 127 if actor.type is 'enemy' else 255
        return tensor(state, dtype=float).unsqueeze(0)


class Epsilon:
    def __init__(self, max_epochs):
        self._random = round(0.10 * max_epochs)
        self._greedy = round(0.01 * max_epochs)
        self._max_epochs = max_epochs - self._random - self._greedy
        self._count = 0
        self.epsilon = 1

    def step(self):
        if self._random <= self._count < self._max_epochs + self._random:
            self.epsilon = 1 - (self._count - self._random) ** 2 / self._max_epochs ** 2
        elif self._count > self._max_epochs + self._random:
            self.epsilon = 0
        self._count += 1
        return self.epsilon


class Policy:
    def __init__(self, memory_size, batch_size, gamma=0.9, epsilon=0):
        self.memory = ReplayMemory(memory_size, batch_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = None
        self.actions = None
        self.policy_net = None
        self.optimizer = None
        self.loss = Loss()

    def set_world_properties(self, n_actions):
        self.n_actions = n_actions
        self.actions = range(self.n_actions)
        self.policy_net = DQN(outputs=self.n_actions)
        # noinspection PyUnresolvedReferences
        self.optimizer = Optimizer(self.policy_net.parameters())

    def optimize(self, transition):
        self.memory.push(transition)
        transitions = self.memory.get_batch()
        batch = Transition(*zip(*transitions))
        curr_state_batch = stack(batch.curr_state)
        prev_state_batch = stack(batch.prev_state)
        last_action_batch = stack(batch.last_action).unsqueeze(1)
        reward_batch = stack(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the
        # actions which would've been taken for each batch state according to policy_net.
        state_action_values = self.policy_net(prev_state_batch).gather(1, last_action_batch).squeeze(1)
        # Compute V(s_{t+1}) for all current states. Expected values of actions for curr_state_batch are computed based
        # on the policy_net; selecting their best reward with max(1)[0].
        state_values = self.policy_net(curr_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (state_values * self.gamma) + reward_batch
        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def strategy(self, state):
        # Explore or Exploit
        return choice(self.actions) if self.epsilon > random() \
            else self.policy_net(state.unsqueeze(0)).max(1)[1].detach()


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = [None] * capacity
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def get_batch(self):
        return sample(self.memory, self.batch_size)

    def push(self, transition):
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


class DQN(Module):
    def __init__(self, outputs=5):
        super(DQN, self).__init__()
        self.conv1 = Conv2d(1, 8, kernel_size=3)
        self.bn1 = BatchNorm2d(8)
        self.head = Linear(72, outputs)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        return self.head(x.view(x.size(0), -1))


if __name__ == '__main__':
    MAX_EPOCHS = 10000
    PRINT_NUM = MAX_EPOCHS // 100
    GAMMA = 0.95
    MEMORY_SIZE = 64
    BATCH_SIZE = 64

    plot = Plot(MAX_EPOCHS, rolling={'method': 'mean', 'N': PRINT_NUM}, figure_num=0)
    plot_epsilon = Plot(MAX_EPOCHS, title='Epsilon vs Epoch', ylabel='Epsilon', figure_num=1)

    eps = Epsilon(MAX_EPOCHS)
    policy = Policy(MEMORY_SIZE, BATCH_SIZE, GAMMA, eps.epsilon)
    world = World(policy.strategy)
    policy.set_world_properties(len(world.actions))

    # Fill replay memory
    for i in range(MEMORY_SIZE):
        if world.play(silent=True):
            policy.memory.push(world.transition())
        else:
            policy.memory.push(world.transition())
            world.reset()
    world.reset()

    tictoc = TicToc(MAX_EPOCHS)
    for epoch in range(MAX_EPOCHS):
        while world.play(silent=True):
            policy.optimize(world.transition())
        policy.optimize(world.transition())
        plot.update(epoch, world.step_count())
        plot_epsilon.update(epoch, policy.epsilon)
        if (epoch + 1) % PRINT_NUM == 0:
            tictoc.toc()
            print('Epoch %7d;' % (epoch + 1), 'Step count: %5d;' % plot.roll[epoch],
                  'eta (s): %6.2f; ' % tictoc.eta(epoch))
        if not epoch + 1 == MAX_EPOCHS:
            policy.epsilon = eps.step()
            world.reset()

    tictoc.toc()
    print('Max duration: %d;' % max(list(plot.roll.values())[-len(plot.roll) // 2:]),
          'Elapsed %.2f (s)' % tictoc.elapsed())
    plot.plot()
    plot_epsilon.plot()
