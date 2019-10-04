from q_tools import Plot
from random import choice
from time import sleep
from math import exp, log
from typing import List


def new_position(position, action):
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

    def move(self, strategy, check_rule):
        valid = False
        while not valid:
            action = strategy()
            x_new, y_new = new_position(self.get_pos(), action)
            valid = check_rule((x_new, y_new))
            if valid:
                self.action = action
                self.x_prev = self.x
                self.y_prev = self.y
                self.x = x_new
                self.y = y_new

    def transition(self):
        return (self.x_prev, self.y_prev), self.action, (self.x, self.y)


class SimpleBoard:
    def __init__(self, n, m, actors):
        self.n = n
        self.m = m
        self.actors = actors
        self.draw()

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


class Player(Actor):
    def __init__(self, x=0, y=0):
        super().__init__(x, y)
        self.type = 'player'


class Enemy(Actor):
    def __init__(self, x=0, y=0):
        super().__init__(x, y)
        self.type = 'enemy'


class World:
    player: Actor
    enemies: List[Actor]
    actors: List[Actor]

    def __init__(self, model_strategy, n=7, m=7):
        self.actions = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
        self.model_strategy = model_strategy
        self.n = n
        self.m = m
        self.player = Player(1, 1)
        self.enemies = [Enemy(4, 1), Enemy(1, 4), Enemy(4, 4)]
        self.actors = [self.player] + self.enemies
        self.board = SimpleBoard(self.n, self.m, self.actors)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        for actor in self.actors:
            actor.reset()

    def game_over(self):
        return any(map(lambda enemy: enemy.get_pos() == self.player.get_pos(), self.enemies))

    def rule(self, position):
        x, y = position
        return 0 <= x < self.n and 0 <= y < self.m

    def strategy(self):
        return choice(self.actions)

    def draw(self):
        sleep(1)
        self.board.draw()
        print('Steps:', self.step_count)

    def play(self, silent=False):
        self.step_count += 1
        for actor in self.actors:
            actor.move(self.strategy, self.rule)
        if not silent:
            self.draw()
        return not self.game_over()


class Epsilon:
    def __init__(self, max_epochs):
        self.eps_start = exp(log(1e-6)/max_epochs)
        self.eps = 1

    def step(self):
        self.eps *= self.eps_start
        return self.eps if self.eps > 1e-5 else 0


if __name__ == '__main__':
    MAX_EPOCHS = 100
    ALPHA = 0.5
    GAMMA = 0.9

    plot = Plot(rolling={'method': 'mean', 'N': 10})
    eps = Epsilon(MAX_EPOCHS)

    world = World(None)
    for _ in range(MAX_EPOCHS):
        while world.play(silent=True):
            pass
        plot.update(world.step_count)
        world.reset()
