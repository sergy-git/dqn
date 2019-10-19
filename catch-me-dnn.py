from q_tools import *
from random import choice, random, sample
from math import copysign
from time import sleep
from typing import List, Tuple, Callable
from torch import tensor, float, stack, save, load
from torch.nn import Module, ModuleList, Linear, BatchNorm1d
from torch.optim import RMSprop as Optimizer
from torch.nn import SmoothL1Loss as Loss
from torch.nn.functional import relu
from collections import namedtuple
# TODO: Add comments
# TODO: Combine with catch-me
# TODO: Convert main to function with hyper parameters
# TODO: Add run-me.py to run all
# TODO: (optional) Add .json file for run settings


class Actor:
    """Actor class for 2D-board"""
    def __init__(self, x: int, y: int):
        """
        :param x: actor initial x position index
        :param y: actor initial y position index
        """
        # Allows Actor's reset with preserving initial (x, y) position index
        self._init_x = x
        self._init_y = y

        # Set initial (x, y) position index
        self._x = x
        self._y = y

        # Prepare empty variables for full transition description
        self._x_prev = None
        self._y_prev = None
        self._last_action = None

        # Prepare empty variable for Actor type description
        self.type = None

    def reset(self, x: int = None, y: int = None):
        """
        Actor's reset, with preserving initial (x, y) position index if x & y not defined
        :param x: Change initial x (optional)
        :param y: Change initial y (optional)
        """
        # Preserve initial (x, y) position index
        self._x = self._init_x if x is None else x
        self._y = self._init_y if y is None else y

        # Clear full transition description
        self._x_prev = None
        self._y_prev = None
        self._last_action = None

    def curr_pos(self) -> (int, int):
        """
        Get current position
        :return: (x, y) current position tuple
        """
        return self._x, self._y

    def last_action(self) -> (int, int):
        """
        Get last action value
        :return: last action (x, y) tuple
        """
        return self._last_action

    def prev_pos(self) -> (int, int):
        """
        Get previous position
        :return: (x, y) previous position tuple
        """
        return self._x_prev, self._y_prev

    def next_pos(self, action: (int, int)) -> (int, int):
        """
        Calculates next position for specific action, adds action (x, y) values to current position (x, y)
        :param action: (x, y) tuple
        :return: (x, y) next position tuple
        """
        return self._x + action[0], self._y + action[1]

    def move(self, action: (int, int), valid_pos: Callable[[Tuple[int, int]], bool]):
        """
        Perform single move of an Actor according to action & verify move validity. If invalid stay on same position
        :param action: (x, y) tuple
        :param valid_pos: external function that verify position validity, valid_pos(position: (int, int)) -> bool
        """
        # Verify new position validity, if invalid stay on same position
        if not valid_pos(self.next_pos(action)):
            action = (0, 0)

        # Perform step
        self.step(action)

    def step(self,  action: (int, int)):
        """
        Perform single step of an Actor according to input action
        :param action: (x, y) tuple
        """
        # Perform move and set new full transition description: previous position, action, current position
        x_new, y_new = self.next_pos(action)
        self._last_action = action
        self._x_prev = self._x
        self._y_prev = self._y
        self._x = x_new
        self._y = y_new


class Player(Actor):
    """Actor from type Player"""
    def __init__(self, x: int, y: int):
        """
        :param x: player initial x position index
        :param y: player initial y position index
        """
        # Initiate super class
        super().__init__(x, y)
        # Set actor type
        self.type = 'player'


class Enemy(Actor):
    """Actor from type Enemy"""
    def __init__(self, x: int, y: int):
        # Initiate super class
        super().__init__(x, y)
        # Set actor type
        self.type = 'enemy'

    def move_smart(self, player_position):
        px, py = player_position
        ex, ey = self.curr_pos()
        dx = px - ex
        dy = py - ey
        vx = abs(dx)
        vy = abs(dy)
        sx = int(copysign(1, dx)) if not dx == 0 else 0
        sy = int(copysign(1, dy)) if not dy == 0 else 0
        if vx > vy:
            action = (sx, 0)
        elif vx < vy:
            action = (0, sy)
        else:
            action = choice([(sx, 0), (0, sy)])

        # Perform step, there is no need to verify new position validity
        self.step(action)


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
    enemies: List[Enemy]
    actors: List[Actor]

    # Full transition definition
    Transition = namedtuple('Transition', ('prev_state', 'last_action', 'curr_state', 'reward'))

    def __init__(self, policy_strategy, n=5, m=5):
        self.actions = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
        self.keys = {self.actions[index]: index for index in range(len(self.actions))}
        self.policy = policy_strategy
        self.n = n
        self.m = m
        self.player = Player(2, 2)
        self.enemies = [Enemy(0, 4), Enemy(4, 0)]
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
        return self.Transition(self.prev_state(), self.last_action(), self.curr_state(), self.reward())

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
            self._reward = -2
        elif self._last_action == (0, 0):
            self._reward = -1
        else:
            self._reward = 1
        return game_over

    def random_action(self):
        return choice(self.actions)

    def strategy(self):
        return self.actions[self.policy(self.state())]

    def play(self, silent=False, smart_enemy=True):
        self.player.move(self.strategy(), self.valid_pos)
        for enemy in self.enemies:
            if smart_enemy:
                enemy.move_smart(self.player.curr_pos())
            else:
                enemy.move(self.random_action(), self.valid_pos)
        self._prev_state = self._curr_state
        self._last_action = self.player.last_action()
        self._curr_state = self.state()
        self._step_count += 1
        if not silent:
            self.draw()
        return not self.game_over()

    def state(self):
        state = [[10] * self.n for _ in range(self.m)]
        for actor in self.actors:
            x, y = actor.curr_pos()
            state[y][x] = 127 if actor.type is 'enemy' else 255
        return tensor(state, dtype=float).unsqueeze(0)


class Epsilon:
    def __init__(self, max_epochs, p_random=.01, p_greedy=.50, greedy_min=1e-6):
        self._random = round(p_random * max_epochs)
        self._greedy = round(p_greedy * max_epochs)
        self._max_epochs = max_epochs - self._random - self._greedy
        self._greedy_min = greedy_min
        self._count = 0
        self.epsilon = 1

    def step(self):
        if self._random <= self._count < self._max_epochs + self._random:
            self.epsilon = 1 - (self._count - self._random) ** 2 / self._max_epochs ** 2
        elif self._count > self._max_epochs + self._random:
            self.epsilon = self._greedy_min
        self._count += 1
        return self.epsilon


class Policy:
    def __init__(self, memory_size, gamma=0.9, epsilon=0):
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = None
        self.net = None
        self.optimizer = None
        self.Transition = None
        self.loss = Loss()
        self._rolling_loss = 0
        self._counts = 0

    def set_world_properties(self, transition, n_actions, n, m):
        self.Transition = transition
        self.actions = range(n_actions)
        self.net = DQN(inputs=n * m, outputs=n_actions)
        # noinspection PyUnresolvedReferences
        self.optimizer = Optimizer(self.net.parameters())

    def optimize(self, batch_size, random_batch=True):
        transitions = self.memory.get_batch(batch_size) if random_batch else self.memory.get_last(batch_size)
        if transitions is not None:
            batch = self.Transition(*zip(*transitions))
            curr_state_batch = stack(batch.curr_state)
            prev_state_batch = stack(batch.prev_state)
            last_action_batch = stack(batch.last_action).unsqueeze(1)
            reward_batch = stack(batch.reward).unsqueeze(1)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These
            # are the actions which would've been taken for each batch state according to policy_net.
            state_action_values = self.net(prev_state_batch).gather(1, last_action_batch)

            # Compute V(s_{t+1}) for all current states. Expected values of actions for curr_state_batch are
            # computed based on the policy_net; selecting their best reward with max(1)[0].
            state_values = self.net(curr_state_batch).max(1)[0].unsqueeze(1).detach()

            # Compute the expected Q values (target)
            expected_state_action_values = (state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = self.loss(state_action_values, expected_state_action_values)
            self._rolling_loss += loss.detach()
            self._counts += 1

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def strategy(self, state):
        # Explore or Exploit
        if self.epsilon > random():
            action = choice(self.actions)
        else:
            self.net.eval()
            action = self.net(state.unsqueeze(0)).max(1)[1].detach()
            self.net.train()

        return action

    def save(self, path):
        save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(load(path))
        self.net.eval()

    def rolling_loss(self, reset=True):
        rolling_loss = self._rolling_loss / self._counts if not self._counts == 0 else None
        if reset:
            self._rolling_loss = 0
            self._counts = 0
        return rolling_loss


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def get_batch(self, batch_size):
        return sample(self.memory, batch_size) if len(self) >= batch_size else None

    def get_last(self, batch_size):
        if batch_size > self.capacity:
            raise ValueError
        position = 0 if self.position >= batch_size else self.position
        return self.memory[-(batch_size - position):] + self.memory[:position] if len(self) >= batch_size else None

    def push(self, transition):
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


class DQN(Module):

    def __init__(self, inputs=25, hidden_depth=1, hidden_dim=16, outputs=5):
        super(DQN, self).__init__()
        self._input = Linear(inputs, hidden_dim)

        self._hidden_B = ModuleList([BatchNorm1d(hidden_dim) for _ in range(hidden_depth)])
        self._hidden_L = ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth)])
        self._output = Linear(hidden_dim, outputs)

    def forward(self, x):
        x = relu(self._input(x.view(x.size(0), -1)))
        # noinspection PyTypeChecker
        for batch_norm, linear in zip(self._hidden_B, self._hidden_L):
            x = relu(batch_norm(linear(x)))
        return self._output(x)


if __name__ == '__main__':
    MAX_EPOCHS = 1500
    PRINT_NUM = 100
    GAMMA = 0.999
    MEMORY_SIZE = 128
    BATCH_SIZE = 64
    NET_PATH = './mem/policy_net.pkl'

    plot = Plot(MAX_EPOCHS, rolling={'method': 'mean', 'N': PRINT_NUM}, figure_num=0)
    plot_epsilon = Plot(MAX_EPOCHS, title='Epsilon vs Epoch', ylabel='Epsilon', figure_num=1)
    plot_loss = Plot(MAX_EPOCHS, title='Loss vs Epoch', ylabel='Loss', figure_num=2,
                     rolling={'method': 'mean', 'N': PRINT_NUM})

    eps = Epsilon(max_epochs=MAX_EPOCHS, p_random=0.1, p_greedy=0.1, greedy_min=1e-4)
    policy = Policy(MEMORY_SIZE, GAMMA, eps.epsilon)
    world = World(policy.strategy)
    policy.set_world_properties(world.Transition, len(world.actions), world.n, world.m)

    tictoc = TicToc(MAX_EPOCHS)
    for epoch in range(MAX_EPOCHS):
        while world.play(silent=True, smart_enemy=False):
            policy.memory.push(world.transition())
        policy.memory.push(world.transition())
        policy.optimize(BATCH_SIZE, random_batch=True)
        plot.update(epoch, world.step_count())
        plot_epsilon.update(epoch, policy.epsilon)
        plot_loss.update(epoch, policy.rolling_loss())
        if (epoch + 1) % PRINT_NUM == 0:
            tictoc.toc()
            print('Epoch %7d;' % (epoch + 1), 'Step count: %5d;' % plot.roll[epoch],
                  'loss: %7.3f; ' % plot_loss.roll[epoch], 'eta (s): %6.2f; ' % tictoc.eta(epoch))

        if not epoch + 1 == MAX_EPOCHS:
            policy.epsilon = eps.step()
            world.reset()

    policy.save(NET_PATH)
    tictoc.toc()
    print('Max duration: %d;' % max(list(plot.roll.values())[-len(plot.roll) // 2:]),
          'Elapsed %.2f (s)' % tictoc.elapsed())
    plot.plot()
    plot_epsilon.plot()
    plot_loss.plot()

    # policy.epsilon = 0
    # world.reset()
    # while world.play(silent=False, smart_enemy=True):
    #     pass
