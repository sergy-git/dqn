from time import sleep
from random import choice, random
from math import exp, log
from q_tools import Plot


# Test
def new_position(position, action):
    return position[0] + action[0], position[1] + action[1]


class SimpleBoard:
    def __init__(self, n, m, cliff, player):
        self.n = n
        self.m = m
        self.cliff = cliff
        self.player = player
        self.draw()

    def draw(self):
        x, y = self.player.get_pos()
        rows = [['[ ]'] * self.n for _ in range(self.m)]
        rows[0][0] = '[#]'
        rows[0][-1] = '[$]'
        for idx in range(self.n - 2):
            rows[0][1 + idx] = '   '
        if (x, y) in self.cliff:
            rows[y][x] = ' x '
        else:
            rows[y][x] = '[x]'
        print('\n' * 5)  # Clear screen
        for row in rows:
            for cell in row:
                print(cell, end='')
            print()
        print('___' * self.n)


class Player:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.test_action = None

    def reset(self):
        self.x = 0
        self.y = 0
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.test_action = None

    def get_pos(self):
        return self.x, self.y

    def set_pos(self):
        self.action = self.test_action
        self.x_prev = self.x
        self.y_prev = self.y
        self.x += self.action[0]
        self.y += self.action[1]

    def tst_act(self, strategy, rules):
        self.test_action = strategy(self.get_pos())
        return rules(new_position(self.get_pos(), self.test_action))

    def state(self):
        return (self.x_prev, self.y_prev), self.action, (self.x, self.y)


class World:
    def __init__(self, model_strategy, model_optim, n=11, m=4):
        self.model_strategy = model_strategy
        self.model_optim = model_optim
        self.n = n
        self.m = m
        self.actions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        self.player = Player()
        self.step_count = 0
        self.cliff = None
        self.board = None
        self.win = None
        self.game_over = None
        self.init_cliff()

    def reset(self):
        self.player.reset()
        self.step_count = 0
        self.win = None
        self.game_over = None

    def init_cliff(self):
        tmp = []
        for p in range(1, self.n - 1):
            tmp.append((p, 0))
        self.cliff = tuple(tmp)

    def in_board(self, position):
        x, y = position
        return 0 <= x < self.n and 0 <= y < self.m

    def init_board(self):
        self.board = SimpleBoard(self.n, self.m, self.cliff, self.player)

    def draw_board(self):
        self.board.draw()
        print('State:', self.state(), '; Reward:', self.reward())

    def step(self, silent=False):
        if not silent and self.board is None:
            self.init_board()

        while not self.player.tst_act(self.strategy, self.in_board):
            pass
        self.player.set_pos()

        if not silent:
            sleep(.5)
            self.draw_board()

        self.step_count += 1
        self.check_game_over()
        return not self.game_over

    def check_game_over(self):
        x, y = self.player.get_pos()
        self.win = x == self.n - 1 and y == 0
        self.game_over = self.win or (x, y) in self.cliff

    def reward(self):
        if self.win:
            return 0
        elif self.game_over:
            return -100
        else:
            return -1

    def state(self):
        return self.player.state()

    def play(self):
        while self.step():
            pass
        self.draw_board()
        print('Win,' if self.win else 'Lose,', 'steps =', self.step_count)

    def strategy(self, position):
        return self.model_strategy(position, self.actions)

    def optim(self):
        self.model_optim(self.state(), self.reward(), self.in_board, self.actions)


class QTable:
    def __init__(self, default_value=0):
        self.default = default_value
        self.dictionary = {(0, 0): {(0, 1): self.default, (1, 0): self.default}}

    def __len__(self):
        return len(self.dictionary)

    def verify(self, position, rules, actions):
        if position not in self.dictionary.keys():
            self.dictionary.update({position: {}})
            for action in actions:
                if rules(new_position(position, action)):
                    self.dictionary[position].update({action: self.default})

    def add(self, position, action, update):
        self.dictionary[position][action] += update

    def get(self, position, action):
        return self.dictionary[position][action]

    def get_max(self, position):
        return max(self.dictionary[position], key=self.dictionary[position].get)

    def get_max_val(self, position):
        return self.dictionary[position][self.get_max(position)]


class QModel:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.Q = QTable()
        self.eps = None

    def set_eps(self, eps_value=0):
        self.eps = eps_value

    def optim(self, state, reward, rules, actions):
        prev_position, action, position = state
        self.Q.verify(position, rules, actions)

        target = reward + self.gamma * self.Q.get_max_val(position)
        error = target - self.Q.get(prev_position, action)
        self.Q.add(prev_position, action, self.alpha * error)

    def strategy(self, position, actions):
        return choice(actions) if self.eps > random() else self.Q.get_max(position)


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

    plot = Plot()
    eps = Epsilon(MAX_EPOCHS)
    model = QModel(ALPHA, GAMMA)

    world = World(model.strategy, model.optim)
    for _ in range(MAX_EPOCHS):
        model.set_eps(eps.step())
        while world.step(silent=True):
            world.optim()
        world.optim()
        plot.update(world.step_count)
        world.reset()

    model.set_eps()
    world.play()
