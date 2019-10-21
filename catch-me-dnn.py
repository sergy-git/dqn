from q_tools import *
from random import choice, random, sample
from math import copysign
from functools import reduce
from time import sleep
from typing import List, Tuple, Callable, Dict, TypeVar, NamedTuple
from torch import tensor, float, stack, save, load, Tensor
from torch.nn import Module, ModuleList, Linear, BatchNorm1d
from torch.optim import RMSprop as Optimizer
from torch.nn import SmoothL1Loss as Loss
from torch.nn.functional import relu

# TODO: Add comments
# TODO: Combine with catch-me
# TODO: Convert main to function with hyper parameters
# TODO: Add run-me.py to run all
# TODO: (optional) Add .json file for settings

# Position definition: (x, y) position index
Position = Tuple[int, int]
# Action definition, (dx, dy) addition to (x, y) position index
Action = Tuple[int, int]
# State definition, tensor or tuple
State = TypeVar('State', tuple, Tensor)
# Last action key index
LastAction = TypeVar('LastAction', int, Tensor)
# Immediate reward value
Reward = TypeVar('Reward', int, Tensor)


class Transition(NamedTuple):
    """Full transition definition"""
    prev_state: State
    last_action: LastAction
    curr_state: State
    reward: Reward


class Actor:
    """Actor class for 2D-board"""
    _x: int  # Actor x position index
    _y: int  # Actor y position index
    _init_x: int  # Actor initial x position index
    _init_y: int  # Actor initial y position index
    _x_prev: int  # Actor previous x position index
    _y_prev: int  # Actor previous y position index
    _last_action: Action  # Last action performed

    def __init__(self, x: int, y: int):
        """
        :param x: actor initial x position index
        :param y: actor initial y position index
        """
        # Prepare empty variable for Actor type description
        self.type = None

        # Actor's reset saving initial (x, y) position index
        self.reset(x, y)

    def reset(self, x: int = None, y: int = None):
        """
        Actor's reset, with preserving initial (x, y) position index, if x & y not defined
        :param x: Change initial x (optional)
        :param y: Change initial y (optional)
        """
        # Preserve initial (x, y) position index, if new x & y not selected
        self._init_x = self._init_x if x is None else x
        self._init_y = self._init_y if y is None else y
        self._x = self._init_x
        self._y = self._init_y

        # Clear full transition description
        self._x_prev = self._init_x
        self._y_prev = self._init_y
        self._last_action = (0, 0)

    def curr_pos(self) -> Position:
        """
        Get current position
        :return: (x, y) current position tuple
        """
        return self._x, self._y

    def last_action(self) -> Action:
        """
        Get last action value
        :return: last action (x, y) tuple
        """
        return self._last_action

    def prev_pos(self) -> Position:
        """
        Get previous position
        :return: (x, y) previous position tuple
        """
        return self._x_prev, self._y_prev

    def next_pos(self, action: Action) -> Position:
        """
        Calculates next position for specific action, adds action (x, y) values to current position (x, y)
        :param action: (x, y) tuple
        :return: (x, y) next position tuple
        """
        return self._x + action[0], self._y + action[1]

    def _step(self, action: Action):
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

    def move(self, action: Action, valid_pos: Callable[[Position], bool]):
        """
        Perform single move of an Actor according to action & verify move validity. If invalid stay on same position
        :param action: (x, y) tuple
        :param valid_pos: external function that verify position validity, valid_pos(position: (int, int)) -> bool
        """
        # Verify new position validity, if invalid stay on same position
        if not valid_pos(self.next_pos(action)):
            action = (0, 0)

        # Perform step
        self._step(action)


class Player(Actor):
    """Actor from type Player"""
    type: str  # Actor type

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
    type: str  # Actor type

    def __init__(self, x: int, y: int):
        # Initiate super class
        super().__init__(x, y)
        # Set actor type
        self.type = 'enemy'

    def move_smart(self, player_position: Position):
        """
        Close gap between the Enemy and the Player
        :param player_position: Players (x, y) position index, external information
        """
        # Get player position & self position
        px, py = player_position
        ex, ey = self.curr_pos()

        # Calculate best action, evaluate it's absolute value and sign
        dx = px - ex
        dy = py - ey
        vx = abs(dx)
        vy = abs(dy)
        sx = int(copysign(1, dx)) if not dx == 0 else 0
        sy = int(copysign(1, dy)) if not dy == 0 else 0

        # Choose best action to catch the Player
        if vx > vy:
            # The gap along x-axis is greater, close gap along x-axis
            action = (sx, 0)
        elif vx < vy:
            # The gap along y-axis is greater, close gap along y-axis
            action = (0, sy)
        else:
            # The gaps along x-axis & y-axis are same, chose randomly
            action = choice([(sx, 0), (0, sy)])

        # Perform step, there is no need to verify new position validity
        self._step(action)


class Board2D:
    """Class for 2D-board"""
    _n: int  # Board width
    _m: int  # Board height
    _board: List[List[str]]  # m * n list of board cells

    def __init__(self, n: int, m: int):
        """
        Initiate 2D board, with origin at upper-left corner
        :param n: number of cells on x-axis, width
        :param m: number of cells on y-axis, height
        """
        self._n = n
        self._m = m
        self._board = self._clear_board()

    def _clear_board(self) -> List[List[str]]:
        """
        Prepare 2D board with empty cells
        :return: 2D board m * n, of type: [['[ ]', ... , '[ ]'], ... , ['[ ]', ... , '[ ]']]
        """
        return [['[ ]'] * self._n for _ in range(self._m)]

    def _draw(self, screen_height: int):
        """
        Clear screen and print 2D board
        :param screen_height: Number of <LF> required to clear the screen
        """
        # Clear screen
        print('\n' * screen_height)
        # Print 2D board
        for row in self._board:
            for cell in row:
                print(cell, end='')
            print()
        print('___' * self._n)


class CatchMeBoard(Board2D):
    """Class for catch-me board"""
    _actors: List[Actor]  # List of actors, player and enemies

    def __init__(self, n: int, m: int, actors: List[Actor]):
        """
        Initiate 2D board, with origin at upper-left corner, with Actors
        :param n: number of cells on x-axis, width
        :param m: number of cells on y-axis, height
        :param actors: list of Actors, Player & Enemies
        """
        super().__init__(n, m)
        self._actors = actors

    def draw(self, screen_height: int = 5):
        """
        Clear screen, mark Actors & print 2D board
        :param screen_height: Number of <LF> required to clear the screen
        """
        # Clear screen
        self._board = self._clear_board()

        # Mark Actors
        for actor in self._actors:
            x, y = actor.curr_pos()
            self._board[y][x] = '[x]' if actor.type is 'enemy' else '[@]'

        # Print 2D board
        self._draw(screen_height)


class World:
    """Class describes world properties and includes: 2D board, actors (player and enemies) and player's strategy"""
    # Available set of actions definition
    actions = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
    # Set State type, 'tuple' or 'tensor'
    dtype = 'tensor'

    _action_index: Dict[Action, int]    # LUT - converts action to index
    _policy: Callable[[State], int]     # policy function that return player's action_index for given state
    _rx: range                          # range of indexes along x-axis, width
    _ry: range                          # range of indexes along y-axis, height
    _player: Actor                      # pointer to player object
    _enemies: List[Enemy]               # pointer to enemies objects list
    _actors: List[Actor]                # pointer to all world actors objects list, player and enemies
    _board: CatchMeBoard                # 2D board object
    _step_count: int                    # counts steps in current Play
    _prev_state: State                  # previous state
    _last_action: Tuple[int, int]       # last action performed
    _curr_state: State                  # current state
    _reward: int                        # immediate reward value

    def __init__(self, player_policy: Callable[[State], int], n: int = 5, m: int = 5, n_enemies: int = 2):
        """
        :param player_policy: policy function that return player's action_index for given state
        :param n: number of cells along x-axis, width
        :param m: number of cells along y-axis, height
        :param n_enemies: number of enemies
        """
        # Evaluate action indexes
        self._action_index = {self.actions[index]: index for index in range(len(self.actions))}
        # Set player's movement policy
        self._policy = player_policy
        # Set x & y axis ranges
        self._rx = range(n)
        self._ry = range(m)
        # Initiate n_actors at (0, 0) position
        self._init_actors(n_enemies)
        # Reset actors to random positions
        self.reset()
        # Set Board Object at size n, m and link Actors to board
        self._board = CatchMeBoard(n, m, self._actors)

    def _init_actors(self, n: int):
        """
        Initiate all Actors, Player and Enemies at (0, 0) position
        :param n:  number of enemies, n >= 1
        """
        # Verify minimal number of actors is 1
        n = 1 if n < 1 else n
        # Initiate Player
        self._player = Player(0, 0)
        # Initiate Enemies
        self._enemies = [Enemy(0, 0) for _ in range(n)]
        # Prepare full Actors list
        self._actors = [self._player] + self._enemies

    def _state(self) -> State:
        """
        State definition: state is a board image, when every pixel is cell
        :return: Return world state
        """
        # Prepare empty 'gray' board
        state = [[127 for _ in self._rx] for _ in self._ry]
        # Set 'black' for player and 'white' for enemies
        for actor in self._actors:
            x, y = actor.curr_pos()
            state[y][x] = 255 if actor.type is 'enemy' else 0

        # Return according to data type
        if self.dtype is 'tensor':
            return tensor(state, dtype=float).unsqueeze(0)
        elif self.dtype is 'tuple':
            return tuple(reduce(lambda a, b: a + b, state))
        else:
            raise ValueError("Unknown type %s." % self.dtype)

    def _draw(self):
        """Draw world current state (board)"""
        sleep(1)                            # some delay to enable movement recognition
        self._board.draw()                  # draw board
        print('Steps:', self._step_count)   # print current step counter value

    def _game_over(self) -> bool:
        """
        Verify game status and update immediate reward value
        :return: true if game over
        """
        # If any enemy on same position with player the game is over
        game_over = any(map(lambda enemy: enemy.curr_pos() == self._player.curr_pos(), self._enemies))

        # Update reward value
        if game_over:
            # reward for game over
            self._reward = -2
        elif self._last_action == (0, 0):
            # reward for not moving, while game not over
            self._reward = -1
        else:
            # reward for making a move, while game not over
            self._reward = 1

        return game_over

    def next_action(self) -> Action:
        """
        Player's next action according to policy and current state
        :return: player's next action
        """
        return self.actions[self._policy(self._curr_state)]

    def valid_pos(self, position: Position) -> bool:
        """
        Validate that position is in the board
        :param position: (x, y) position index tuple
        :return: True if position in side the board
        """
        x, y = position
        return x in self._rx and y in self._ry

    def reset(self):
        """Reset world parameters: clear step counter, set actors to random positions and evaluate initial transition"""
        # Clear step counter
        self._step_count = 0

        # Choose random player position
        self._player.reset(choice(self._rx), choice(self._ry))

        # Choose random enemies positions, different from players position
        for enemy in self._enemies:
            enemy.reset(choice(self._rx), choice(self._ry))
            while self._player.curr_pos() == enemy.curr_pos():
                enemy.reset(choice(self._rx), choice(self._ry))

        # Save initial full transition description
        self._prev_state = self._state()
        self._curr_state = self._prev_state.copy()
        self._last_action = (0, 0)
        self._reward = 0

    def step_num(self) -> int:
        """
        Get number of steps in current Play
        :return: number of steps
        """
        return self._step_count

    def prev_state(self) -> State:
        """
        Get world previous state
        :return: previous state
        """
        return self._prev_state

    def last_action(self) -> LastAction:
        """
        Get player's last action
        :return: last action
        """
        # Return according to data type
        if self.dtype is 'tensor':
            return tensor(self._action_index[self._last_action])
        elif self.dtype is 'tuple':
            return self._action_index[self._last_action]
        else:
            raise ValueError("Unknown type %s." % self.dtype)

    def curr_state(self) -> State:
        """
        Get world current state
        :return: current state
        """
        return self._curr_state

    def reward(self) -> Reward:
        """
        Get immediate reward value
        :return: immediate reward
        """
        # Return according to data type
        if self.dtype is 'tensor':
            return tensor(self._reward, dtype=float)
        elif self.dtype is 'tuple':
            return self._reward
        else:
            raise ValueError("Unknown type %s." % self.dtype)

    def transition(self) -> Transition:
        """
        Full transition description: previous state, last player's action, current state, immediate reward value
        :return: transition
        """
        return Transition(self.prev_state(), self.last_action(), self.curr_state(), self.reward())

    def play(self, silent: bool = False, smart_enemy: bool = True) -> bool:
        """
        Perform single world step
        :param silent: silent mode flag. if true, don't draw world state
        :param smart_enemy: if false, enemies performs random moves
        :return: true if game not over
        """
        # player's step. must be first, because depends on current enemies positions
        self._player.move(self.next_action(), self.valid_pos)
        # enemies step
        for enemy in self._enemies:
            if smart_enemy:
                # close gap to the player
                enemy.move_smart(self._player.curr_pos())
            else:
                # perform random move
                enemy.move(choice(self.actions), self.valid_pos)

        # Save updated full transition description
        self._prev_state = self._curr_state.copy()
        self._last_action = self._player.last_action()
        self._curr_state = self._state()

        # update step counter
        self._step_count += 1
        # draw world state, if not in silent mode
        if not silent:
            self._draw()

        return not self._game_over()


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
            self.optimizer._step()

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
    _policy = Policy(MEMORY_SIZE, GAMMA, eps.epsilon)
    world = World(_policy.strategy)
    _policy.set_world_properties(Transition, len(world.actions), world._rx, world._ry)

    tictoc = TicToc(MAX_EPOCHS)
    for epoch in range(MAX_EPOCHS):
        while world.play(silent=True, smart_enemy=False):
            _policy.memory.push(world.transition())
        _policy.memory.push(world.transition())
        _policy.optimize(BATCH_SIZE, random_batch=True)
        plot.update(epoch, world.step_num())
        plot_epsilon.update(epoch, _policy.epsilon)
        plot_loss.update(epoch, _policy.rolling_loss())
        if (epoch + 1) % PRINT_NUM == 0:
            tictoc.toc()
            print('Epoch %7d;' % (epoch + 1), 'Step count: %5d;' % plot.roll[epoch],
                  'loss: %7.3f; ' % plot_loss.roll[epoch], 'eta (s): %6.2f; ' % tictoc.eta(epoch))

        if not epoch + 1 == MAX_EPOCHS:
            _policy.epsilon = eps.step()
            world.reset()

    _policy.save(NET_PATH)
    tictoc.toc()
    print('Max duration: %d;' % max(list(plot.roll.values())[-len(plot.roll) // 2:]),
          'Elapsed %.2f (s)' % tictoc.elapsed())
    plot.plot()
    plot_epsilon.plot()
    plot_loss.plot()

    _policy.epsilon = 0
    world.reset()
    while world.play(silent=False, smart_enemy=True):
        pass
