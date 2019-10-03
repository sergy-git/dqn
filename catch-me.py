def new_position(position, action):
    return position[0] + action[0], position[1] + action[1]


class Actor:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_prev = None
        self.y_prev = None
        self.action = None
        self.test_action = None
        self.type = None

    def reset(self):
        self.x = 0
        self.y = 0
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


if __name__ == '__main__':
    pass
