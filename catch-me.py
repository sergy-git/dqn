from random import choice

def new_position(position, action):
    return position[0] + action[0], position[1] + action[1]


class Actor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.test_action = None
        self.type = None

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.test_action = None

    def get_pos(self):
        return self.x, self.y

    def move(self):
        self.action = self.test_action
        self.x_prev = self.x
        self.y_prev = self.y
        self.x += self.action[0]
        self.y += self.action[1]

    def valid_move(self, action, rules):
        self.test_action = action
        return rules(new_position(self.get_pos(), self.test_action))

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
    def __init__(self, model_strategy, n=7, m=7):
        self.model_strategy = model_strategy
        self.n = n
        self.m = m
        self.actions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        self.actors = [Player(1, 1), Enemy(3, 1), Enemy(1, 3), Enemy(3, 3)]
        self.board = SimpleBoard(5, 5, self.actors)
        self.step_count = 0
        self.game_over = None

    def game_over(self, player, enemys):
        x, y = player.get_pos()
        for enemy in enemys:
            if enemy


if __name__ == '__main__':
    world = World(choice)

