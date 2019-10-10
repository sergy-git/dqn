from q_tools import *
from random import choice, random
from time import sleep
from typing import List


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
        valid = False
        while not valid:
            action = policy_strategy()
            x_new, y_new = self.next_pos(action)
            valid = valid_pos((x_new, y_new))
            if valid:
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
        return self._reward

    def prev_state(self):
        return self._prev_state

    def last_action(self):
        return self._last_action

    def curr_state(self):
        return self._curr_state

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
            self._reward = 1
        return game_over

    def random_action(self):
        return choice(self.actions)

    def strategy(self):
        return self.policy(self.state(), self.random_action)

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
        t_state = []
        for row in state:
            t_state.append(tuple(row))
        return tuple(t_state)


class Policy:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0, default_value=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default = default_value
        self.q = {}
        self.best_action = {}
        self.actions = None
        self.valid_pos = None
        self.next_pos = None

    def __len__(self):
        return len(self.q)

    def set_world_properties(self, actions, valid_pos):
        self.actions = actions
        self.valid_pos = valid_pos

    def set_actor_properties(self, next_pos):
        self.next_pos = next_pos

    def validate(self, state):
        if state not in self.q.keys():
            self.q.update({state: {}})
            self.best_action.update({state: {'action': (0, 0), 'value': 0}})
            for action in self.actions:
                if self.valid_pos(self.next_pos(action)):
                    self.q[state].update({action: self.default})

    def get(self, state, action):
        return self.q[state][action]

    def state0(self, full_state):
        self.validate(full_state)

    def optimize(self, prev_state, last_action, curr_state, reward):
        self.validate(curr_state)
        target = reward + self.gamma * self.best_action[curr_state]['value']
        error = target - self.q[prev_state][last_action]
        self.q[prev_state][last_action] += self.alpha * error

        if self.q[prev_state][last_action] > self.best_action[prev_state]['value']:
            self.best_action[prev_state]['action'] = last_action
            self.best_action[prev_state]['value'] = self.q[prev_state][last_action]

    def strategy(self, state, random_action):
        return random_action() if self.epsilon > random() else self.best_action[state]['action']


class Epsilon:
    def __init__(self, max_epochs):
        self._random = round(0.1 * max_epochs)
        self._greedy = round(0.1 * max_epochs)
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


if __name__ == '__main__':
    MAX_EPOCHS = 10000
    PRINT_NUM = MAX_EPOCHS // 50
    ALPHA = 0.5
    GAMMA = 0.95

    plot = Plot(MAX_EPOCHS, rolling={'method': 'mean', 'N': PRINT_NUM}, figure_num=0)
    plot_epsilon = Plot(MAX_EPOCHS, title='Epsilon vs Epoch', ylabel='Epsilon', figure_num=1)

    eps = Epsilon(MAX_EPOCHS)
    policy = Policy(ALPHA, GAMMA, eps.epsilon)
    world = World(policy.strategy)
    policy.set_world_properties(world.actions, world.valid_pos)
    policy.set_actor_properties(world.player.next_pos)
    policy.validate(world.curr_state())  # Init state 0 in dictionary

    tictoc = TicToc(MAX_EPOCHS)
    for epoch in range(MAX_EPOCHS):
        while world.play(silent=True):
            policy.optimize(world.prev_state(), world.last_action(), world.curr_state(), world.reward())
        policy.optimize(world.prev_state(), world.last_action(), world.curr_state(), world.reward())
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
