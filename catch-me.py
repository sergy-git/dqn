from q_tools import Plot
from random import choice, random
from time import sleep
from math import exp, log
from typing import List


def next_position(position, action):
    return position[0] + action[0], position[1] + action[1]


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

    def get_pos(self):
        return self.x, self.y

    def prev_pos(self):
        return self.x_prev, self.y_prev

    def next_pos(self, action):
        return next_position(self.get_pos(), action)

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
            x, y = actor.get_pos()
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
        self.empty_state = list(0 for _ in range(n * m))
        self.policy = policy_strategy
        self.n = n
        self.m = m
        self.player = Player(1, 1)
        self.enemies = [Enemy(3, 3)]
        self.actors = [self.player] + self.enemies
        self.board = SimpleBoard(self.n, self.m, self.actors)
        self._step_count = 0
        self._reward = 0

    def reset(self):
        self._step_count = 0
        for actor in self.actors:
            actor.reset()

    def step_count(self):
        return self._step_count

    def game_over(self):
        game_over = any(map(lambda enemy: enemy.get_pos() == self.player.get_pos(), self.enemies))
        if game_over:
            self._reward = -100
        else:
            self._reward = 1
        return game_over

    def valid_pos(self, position):
        x, y = position
        return 0 <= x < self.n and 0 <= y < self.m

    def random_action(self):
        return choice(self.actions)

    def strategy(self):
        return self.policy(self.state(), self.random_action)

    def draw(self):
        sleep(1)
        self.board.draw()
        print('Steps:', self._step_count)

    def play(self, silent=False):
        self._step_count += 1
        self.player.move(self.strategy, self.valid_pos)
        for enemy in self.enemies:
            enemy.move(self.random_action, self.valid_pos)
        if not silent:
            self.draw()
        return not self.game_over()

    def state_update(self, state, position, actor_type):
        x, y = position
        state[x * self.n + y] = -1 if actor_type is 'enemy' else 1
        return state

    def state(self):
        state = self.empty_state.copy()
        for actor in self.actors:
            state = self.state_update(state, actor.get_pos(), actor.type)
        return tuple(state)

    def prev_state(self):
        state = self.empty_state.copy()
        for actor in self.actors:
            state = self.state_update(state, actor.prev_pos(), actor.type)
        return tuple(state)

    def full_state(self):
        return self.state(), self.player.get_pos()

    def transition(self):
        return self.prev_state(), self.player.action, self.full_state()

    def reward(self):
        return self._reward


class Epsilon:
    def __init__(self, max_epochs):
        self.eps_start = exp(log(1e-6)/max_epochs)
        self.epsilon = 1

    def step(self):
        self.epsilon *= self.eps_start
        return self.epsilon if self.epsilon > 1e-5 else 0


class Q:
    def __init__(self, alpha=0.5, default_value=0):
        self.actions = None
        self.valid_pos = None
        self.alpha = alpha
        self.default = default_value
        self.dictionary = {}

    def __len__(self):
        return len(self.dictionary)

    def set_world_rules(self, actions, valid_pos):
        self.actions = actions
        self.valid_pos = valid_pos

    def validate(self, full_sate):
        state, position = full_sate
        if state not in self.dictionary.keys():
            self.dictionary.update({state: {}})
            for action in self.actions:
                if self.valid_pos(next_position(position, action)):
                    self.dictionary[state].update({action: self.default})

    def invalid_state(self, state):
        return state not in self.dictionary.keys()

    def get(self, state, action):
        return self.dictionary[state][action]

    def update(self, state, action, update):
        self.dictionary[state][action] += self.alpha * update

    def best_action(self, state):
        return max(self.dictionary[state], key=self.dictionary[state].get)

    def max_reward(self, state):
        return self.dictionary[state][self.best_action(state)]


class Policy:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0):
        self.gamma = gamma
        self.q = Q(alpha)
        self.epsilon = epsilon

    def set_world_rules(self, actions, valid_pos):
        self.q.set_world_rules(actions, valid_pos)

    def state0(self, full_state):
        self.q.validate(full_state)

    def optimize(self, transition, reward):
        prev_state, action, full_state = transition
        state, _ = full_state

        self.q.validate(full_state)

        target = reward + self.gamma * self.q.max_reward(state)
        error = target - self.q.get(prev_state, action)
        self.q.update(prev_state, action, error)

    def strategy(self, state, random_action):
        return random_action() if self.epsilon > random() or self.q.invalid_state(state) else self.q.best_action(state)


if __name__ == '__main__':
    MAX_EPOCHS = 1000000
    PRINT_NUM = round(MAX_EPOCHS/1000)
    ALPHA = 0.5
    GAMMA = 0.9

    plot = Plot(rolling={'method': 'mean', 'N': 1000})
    eps = Epsilon(MAX_EPOCHS)
    policy = Policy(ALPHA, GAMMA, eps.epsilon)

    world = World(policy.strategy)
    policy.set_world_rules(world.actions, world.valid_pos)
    policy.state0(world.full_state())
    for epoch in range(MAX_EPOCHS):
        while world.play(silent=True):
            policy.optimize(world.transition(), world.reward())
        plot.update(world.step_count(), silent=True)
        if (epoch + 1) % PRINT_NUM == 0:
            print('Epoch %5d;' % (epoch + 1), 'Step count: %4d;' % plot.roll[-1], 'Reward: %d;' % world.reward())
        policy.epsilon = eps.step()
        world.reset()
    plot.plot()
