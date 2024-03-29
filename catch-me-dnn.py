# policy optimization taken from pytorch's REINFORCEMENT LEARNING (DQN) TUTORIAL:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial
# idea for problem and q-learning policy taken from:
# https://habr.com/ru/post/345656/
from q_tools import *
from random import choice, random, sample
from math import copysign
from functools import reduce
from time import sleep
from typing import List, Tuple, Callable, Dict, TypeVar, NamedTuple, Union, Optional
from torch import device, tensor, float, stack, save, load, Tensor
from pickle import dump
from pickle import load as p_load
from torch.nn import Module, ModuleList, Linear, BatchNorm1d
# noinspection PyUnresolvedReferences
from torch.optim import RMSprop as Optimizer
from torch.nn import SmoothL1Loss as Loss
from torch.nn.functional import relu
from numpy import arange
from pandas import DataFrame

# use gpu if there is cuda device
# from torch import cuda
# dev = device('cuda' if cuda.is_available() else 'cpu')

# force cpu
dev = device('cpu')

# position definition: (x, y) position index
Position = Tuple[int, int]
# action definition, (dx, dy) addition to (x, y) position index
Action = Tuple[int, int]
# state definition, tensor or tuple
State = TypeVar('State', Tuple, Tensor)
# last action key index
LastAction = TypeVar('LastAction', int, Tensor)
# immediate reward value
Reward = TypeVar('Reward', int, Tensor)
# definition of optimization algorithm type RQL or DQN, reinforced-q-learning or deep-q-network
AlgoTypes = ['RQL', 'DQN']


class AlgoType:
    """verification of optimization algorithm type RQL or DQN"""
    def __init__(self, dtype: AlgoTypes):
        """
        :param dtype: optimization algorithm type RQL or DQN
        """
        if dtype in AlgoTypes:
            self._type = dtype
        else:
            raise ValueError("Unknown type '%s', expected one from a list: " % dtype, AlgoTypes)

    def item(self):
        """get optimization algorithm type"""
        return self._type


class Transition(NamedTuple):
    """full transition definition"""
    prev_state: State
    last_action: LastAction
    curr_state: State
    reward: Reward


class Actor:
    """actor for 2D-board"""
    _init_x: int            # actor initial x position index
    _init_y: int            # actor initial y position index
    _x_prev: int            # actor previous x position index
    _y_prev: int            # actor previous y position index
    _last_action: Action    # last action performed
    _x: int                 # actor current x position index
    _y: int                 # actor current y position index

    def __init__(self, x: int, y: int):
        """
        :param x: actor initial x position index
        :param y: actor initial y position index
        """
        # prepare empty variable for actor type description
        self.type = None

        # actor's reset saving initial (x, y) position index
        self.reset(x, y)

    def _next_pos(self, action: Action) -> Position:
        """
        calculates next position for specific action, adds action (x, y) values to current position (x, y)
        :param action: (x, y) tuple
        :return: (x, y) next position tuple
        """
        return self._x + action[0], self._y + action[1]

    def _step(self, action: Action):
        """
        perform single step of an actor according to input action
        :param action: (x, y) tuple
        """
        # perform move and set new full transition description: previous position, action, current position
        x_new, y_new = self._next_pos(action)
        self._last_action = action
        self._x_prev = self._x
        self._y_prev = self._y
        self._x = x_new
        self._y = y_new

    def reset(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        actor's reset, with preserving initial (x, y) position index, if x & y not defined
        :param x: Change initial x (optional)
        :param y: Change initial y (optional)
        """
        # preserve initial (x, y) position index, if new x & y not selected
        self._init_x = self._init_x if x is None else x
        self._init_y = self._init_y if y is None else y
        self._x = self._init_x
        self._y = self._init_y

        # clear full transition description
        self._x_prev = self._init_x
        self._y_prev = self._init_y
        self._last_action = (0, 0)

    def prev_pos(self) -> Position:
        """
        get previous position
        :return: (x, y) previous position tuple
        """
        return self._x_prev, self._y_prev

    def last_action(self) -> Action:
        """
        get last action value
        :return: last action (x, y) tuple
        """
        return self._last_action

    def curr_pos(self) -> Position:
        """
        get current position
        :return: (x, y) current position tuple
        """
        return self._x, self._y

    def move(self, action: Action, valid_pos: Callable[[Position], bool]):
        """
        perform single move of an actor according to action & verify move validity. if invalid stay on same position
        :param action: (x, y) tuple
        :param valid_pos: external function that verify position validity, valid_pos(position: (int, int)) -> bool
        """
        # verify new position validity, if invalid stay on same position
        if not valid_pos(self._next_pos(action)):
            action = (0, 0)

        # perform step
        self._step(action)


class Player(Actor):
    """actor from type player"""
    type: str  # actor type

    def __init__(self, x: int = 0, y: int = 0):
        """
        :param x: player initial x position index
        :param y: player initial y position index
        """
        # initiate super class
        super().__init__(x, y)
        # set actor type
        self.type = 'player'


class Enemy(Actor):
    """actor from type enemy"""
    type: str  # actor type

    def __init__(self, x: int = 0, y: int = 0):
        # initiate super class
        super().__init__(x, y)
        # set actor type
        self.type = 'enemy'

    def move_smart(self, player_position: Position):
        """
        close gap between the enemy and the player
        :param player_position: players (x, y) position index, external information
        """
        # get player position & self position
        px, py = player_position
        ex, ey = self.curr_pos()

        # calculate best action, evaluate it's absolute value and sign
        dx = px - ex
        dy = py - ey
        vx = abs(dx)
        vy = abs(dy)
        sx = int(copysign(1, dx)) if not dx == 0 else 0
        sy = int(copysign(1, dy)) if not dy == 0 else 0

        # choose best action to catch the player
        if vx > vy:
            # the gap along x-axis is greater, close gap along x-axis
            action = (sx, 0)
        elif vx < vy:
            # the gap along y-axis is greater, close gap along y-axis
            action = (0, sy)
        else:
            # the gaps along x-axis & y-axis are same, chose randomly
            action = choice([(sx, 0), (0, sy)])

        # perform step, there is no need to verify new position validity
        self._step(action)


class Board2D:
    """2D-board"""
    _n: int  # board width
    _m: int  # board height
    _board: List[List[str]]  # m * n list of board cells

    def __init__(self, n: int, m: int):
        """
        initiate 2D board, with origin at upper-left corner
        :param n: number of cells on x-axis, width
        :param m: number of cells on y-axis, height
        """
        self._n = n
        self._m = m
        self._board = self._clear_board()

    def _clear_board(self) -> List[List[str]]:
        """
        prepare 2D board with empty cells
        :return: 2D board m * n, of type: [['[ ]', ... , '[ ]'], ... , ['[ ]', ... , '[ ]']]
        """
        return [['[ ]'] * self._n for _ in range(self._m)]

    def _draw(self, screen_height: int):
        """
        clear screen and print 2D board
        :param screen_height: number of <LF> required to clear the screen
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
    """catch-me board"""
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


class ReplayMemory:
    """Replay memory stores transitions"""
    _capacity: int                          # maximal memory capacity
    _memory: List[Union[Transition, None]]  # list of memory elements
    _index: int                             # cyclic counter that points to next memory index to fill

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._memory = []
        self._index = 0

    def __len__(self):
        return len(self._memory)

    def random(self, batch_size: int) -> Optional[List[Transition]]:
        """
        get random batch of transitions
        :param batch_size: number of transitions to pass
        :return: batch of transitions
        """
        # return requested batch of transitions if memory already contains enough transitions
        return sample(self._memory, batch_size) if len(self) >= batch_size else None

    def last(self, batch_size: int) -> Optional[List[Transition]]:
        """
        get last, ordered batch of transitions
        :param batch_size: number of transitions to pass
        :return: batch of transitions
        """
        # return requested batch of transitions if memory already contains enough transitions
        if len(self) > batch_size:
            batch_size += 1  # include start position in training, verifies that training batch > 1
            # wrap around verification
            if batch_size > self._index:
                res = self._memory[-(batch_size - self._index):] + self._memory[:self._index]
            else:
                res = self._memory[(self._index - batch_size):self._index]
        else:
            res = None

        return res

    def push(self, transition: Transition) -> None:
        """
        push element to the replay memory
        :param transition: transition
        """
        # enlarge memory if it isn't at maximal capacity
        if len(self) < self._capacity:
            self._memory.append(None)
        self._memory[self._index] = transition
        # enlarge index & wrap around
        self._index = (self._index + 1) % self._capacity


class DQN(Module):
    def __init__(self, inputs: int, outputs: int, hidden_depth: int = 3, hidden_dim: int = 25):
        """
        :param inputs: input layer dimension
        :param hidden_depth: number of hidden layers
        :param hidden_dim: hidden layer dimension
        :param outputs: output layer dimension
        """
        super(DQN, self).__init__()
        # linear input layer
        self._input = Linear(inputs, hidden_dim)
        # list of batch normalizations for hidden layers
        self._hidden_B = ModuleList([BatchNorm1d(hidden_dim) for _ in range(hidden_depth)])
        # list of linear hidden layers
        self._hidden_L = ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth)])
        # linear output layer
        self._output = Linear(hidden_dim, outputs)

    def forward(self, x):
        x = relu(self._input(x.view(x.size(0), -1)))
        # noinspection PyTypeChecker
        for batch_norm, linear in zip(self._hidden_B, self._hidden_L):
            x = relu(batch_norm(linear(x)))
        return self._output(x)


class Epsilon:
    _random: int         # number of exploration epochs at the beginning
    _greedy: int         # number of exploitation epochs at the end
    _max_epochs: int     # maximal epochs, decay algorithm finish with greedy policy
    _explore_min: float  # minimal exploration percent, 0 == greedy
    _item: float         # epsilon current value

    def __init__(self, max_epochs, p_random=.01, p_greedy=.50, explore_min=1e-6):
        """
        :param max_epochs: maximal epochs, decay algorithm finish with greedy policy
        :param p_random: percent of exploration epochs at the beginning
        :param p_greedy: percent of exploitation epochs at the end
        :param explore_min: minimal exploration percent, 0 == greedy
        """
        self._random = round(p_random * max_epochs)
        self._greedy = round(p_greedy * max_epochs)
        self._max_epochs = max_epochs - self._random - self._greedy
        self._explore_min = explore_min
        self._item = 1  # starting exploration percent is 100%

    def decay(self, curr_epoch):
        """
        epsilon decay algorithm, gradually change epsilon from 1 to 0 every epoch, 0 == greedy
        :param curr_epoch: current epoch
        """
        if self._random <= curr_epoch < self._max_epochs + self._random:
            self._item = 1 - (curr_epoch - self._random) ** 2 / self._max_epochs ** 2
        elif curr_epoch > self._max_epochs + self._random:
            self._item = self._explore_min

    def item(self):
        """return current epsilon value"""
        return self._item


class Policy:
    """player's action policy for any state"""
    _memory: ReplayMemory    # cyclic memory block to store transitions history
    _batch_size: int         # number of samples for single optimization step, dnn only
    _lr: float               # learning rate, q-learning only
    _gamma: float            # discount factor
    _alpha: float            # smoothing constant (RMS prop) default = 0.99
    _epsilon_decay: Epsilon  # epsilon decay algorithm
    _epsilon_value: float    # exploration/exploitation percent {0.0 ... 1.0}, 0 == greedy
    _actions: range          # available range of actions
    policy_net: DQN          # policy neural network network, computes V(s_t) expected values of actions for given state
    target_net: DQN          # target neural network network, computes V(s_t) expected values of actions for given state
    _optimizer: Optimizer    # optimizer function
    _q: dict                 # Table for each action value at every state Q(s_t, a)
    _best_action: dict       # best action for each state
    _loss: Loss              # loss function
    _acc_loss: float         # accumulated loss value
    _counts: int             # accumulated loss epoch counter
    _dtype: str              # definition of state type, 'RQL' or 'DQN'

    def __init__(self, memory_size: int, batch_size: int, lr: float = 0.5, gamma: float = 0.9, alpha: float = 0.99,
                 epsilon: Optional[Epsilon] = None, n_hidden: int = 3, w_hidden: int = 25):
        """
        :param memory_size: maximal memory capacity
        :param batch_size: number of samples for single optimization step, dnn only
        :param lr: learning rate, q-learning only
        :param gamma: discount factor, determines the importance of future rewards
        :param epsilon: exploration/exploitation decay algorithm, gradually change epsilon from 1 to 0, 0 == greedy
        """
        self._memory = ReplayMemory(memory_size)
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        self._alpha = alpha
        self._epsilon_decay = epsilon
        self._epsilon_value = self._epsilon_decay.item() if self._epsilon_decay is not None else 0  # greedy if None
        self._n_hidden = n_hidden
        self._w_hidden = w_hidden
        self._loss = Loss()
        self._acc_loss = 0
        self._counts = 0

    def set_world_properties(self, n_actions: int, size: int, dtype: str) -> None:
        """
        initiate parameters that depend on world parameters
        :param n_actions: available number of actions
        :param size: board size
        :param dtype: optimization algorithm type RQL or DQN
        """
        # set action range
        self._actions = range(n_actions)

        # set state data type
        self._dtype = dtype

        # initiate according to data type
        if self._dtype is 'DQN':
            # initiate neural network network, computes V(s_t) expected values of actions for given state
            # noinspection PyUnresolvedReferences
            self.policy_net = DQN(inputs=size, outputs=n_actions,
                                  hidden_depth=self._n_hidden, hidden_dim=self._w_hidden).to(device=dev)
            # noinspection PyUnresolvedReferences
            self.target_net = DQN(inputs=size, outputs=n_actions,
                                  hidden_depth=self._n_hidden, hidden_dim=self._w_hidden).to(device=dev)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            # set optimizer
            # noinspection PyUnresolvedReferences
            self._optimizer = Optimizer(self.policy_net.parameters(), lr=self._lr, alpha=self._alpha)
        elif self._dtype is 'RQL':
            self._q = {}
            self._best_action = {}

    def push(self, transition: Transition) -> None:
        """
        push element to the policy replay memory
        :param transition: transition
        """
        self._memory.push(transition)

    def epsilon_decay(self, curr_epoch):
        """
        update epsilon value according to decay algorithm
        :param curr_epoch: current epoch
        """
        self._epsilon_decay.decay(curr_epoch)
        self._epsilon_value = self._epsilon_decay.item()

    def set_greedy(self):
        """set exploitation policy, greedy"""
        self._epsilon_value = 0

    def eps(self):
        """get epsilon value"""
        return self._epsilon_value

    def optimize(self, batch_size: int, random_batch: Optional[bool] = None) -> None:
        """
        policy optimization step
        :param batch_size: number of transitions to pass
        :param random_batch: use random batch or last game transitions for optimisation
        """
        # return according to data type
        if self._dtype is 'DQN':
            self._optimize_dnn(batch_size=batch_size, random_batch=True if random_batch is None else random_batch)
        elif self._dtype is 'RQL':
            self._optimize_rlq(batch_size=batch_size, random_batch=False if random_batch is None else random_batch)

    def _update_q(self, state):
        """
        add new state to q_table
        :param state: world state - board image, when every pixel is cell
        """
        if state not in self._q.keys():
            self._q.update({state: {}})
            self._best_action.update({state: {'action': 0, 'value': 0}})
            for action in self._actions:
                self._q[state].update({action: 0})

    def _optimize_rlq(self, batch_size: int, random_batch: bool = False) -> None:
        """
        q-learning based policy optimization step
        :param batch_size: number of samples for current optimization step (last game length)
        :param random_batch: use random batch or last game transitions for optimisation
        """
        # load last game batch
        transitions = self._memory.last(batch_size) if not random_batch else self._memory.random(batch_size)

        if transitions is not None:
            for transition in transitions:
                prev_state, last_action, curr_state, reward = transition
                self._update_q(prev_state)
                self._update_q(curr_state)

                target = reward + self._gamma * self._best_action[curr_state]['value']
                error = target - self._q[prev_state][last_action]
                self._q[prev_state][last_action] += self._lr * error
                self._acc_loss += self._lr * error
                self._counts += 1

                if self._q[prev_state][last_action] > self._best_action[prev_state]['value']:
                    self._best_action[prev_state]['action'] = last_action
                    self._best_action[prev_state]['value'] = self._q[prev_state][last_action]

    def _optimize_dnn(self, batch_size: int, random_batch: bool = True) -> None:
        """
        dnn based policy optimization step
        :param batch_size: number of samples for current optimization step
        :param random_batch: use random batch or last game transitions for optimisation
        """
        # load random optimization batch
        transitions = self._memory.random(batch_size) if random_batch else self._memory.last(batch_size)

        if transitions is not None:
            # transpose the batch (https://stackoverflow.com/a/19343/3343043 for detailed explanation). this
            # converts batch-array of transitions to transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            curr_state_batch = stack(batch.curr_state)
            prev_state_batch = stack(batch.prev_state)
            last_action_batch = stack(batch.last_action).unsqueeze(1)
            reward_batch = stack(batch.reward).unsqueeze(1)

            # compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. these
            # are the actions which would've been taken for each batch state according to policy_net.
            state_action_values = self.policy_net(prev_state_batch).gather(1, last_action_batch)

            # compute V(s_{t+1}) for all current states. expected values of actions for curr_state_batch are
            # computed based on the target_net; selecting their best reward with max(1)[0].
            state_values = self.target_net(curr_state_batch).max(1)[0].unsqueeze(1).detach()

            # compute the expected Q values (target)
            expected_state_action_values = (state_values * self._gamma) + reward_batch

            # compute huber loss
            loss = self._loss(state_action_values, expected_state_action_values)
            self._acc_loss += loss.detach().item()
            self._counts += 1

            # optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self._optimizer.step()
            return

    def policy_function(self, state: State) -> int:
        # explore or exploit
        if self._epsilon_value > random():
            # random action
            return choice(self._actions)
        else:
            if self._dtype is 'DQN':
                self.policy_net.eval()    # set to eval() mode, because of batch normalization
                action = self.policy_net(state.unsqueeze(0)).max(1)[1].detach()   # Q(s_t,a) best action for given state
                self.policy_net.train()   # set to train() mode, for optimization
                return action
            elif self._dtype is 'RQL':
                if state in self._best_action.keys():
                    # Q(s_t,a) best action for given state
                    return self._best_action[state]['action']
                else:
                    # random action
                    return choice(self._actions)

    def save(self, path: str) -> None:
        """
        save neural network state dictionary
        :param path: full path for state dictionary file
        """
        if self._dtype is 'DQN':
            save(self.policy_net.state_dict(), path)
        elif self._dtype is 'RQL':
            file = open(path, "wb")
            dump(self._q, file)
            file.close()

    def load(self, path: str) -> None:
        """
        load neural network state dictionary
        :param path: full path for state dictionary file
        """
        if self._dtype is 'DQN':
            self.policy_net.load_state_dict(load(path))
            self.policy_net.eval()
        elif self._dtype is 'DQN':
            file = open(path, 'rb')
            self._q = p_load(file)
            file.close()

    def mean_loss(self, reset: bool = True) -> float:
        """
        calculate mean loss for last epochs
        :param reset: reset accumulated loss & epoch counter
        :return: mean loss value
        """
        # calculate mean loss if epoch counter not zero
        mean_loss = self._acc_loss / self._counts if not self._counts == 0 else None
        if reset:
            self._acc_loss = 0
            self._counts = 0
        return mean_loss


class World:
    """world properties description: 2D board, actors (player and enemies) and player's next move strategy"""
    # definition of available set of actions
    _actions = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))

    _action_index: Dict[Action, int]    # LUT - converts action to index
    _policy: Callable[[State], int]     # policy function that return player's action_index for given state
    _dtype: str                         # definition of state type, 'RQL' or 'DQN'
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

    def __init__(self, player_policy: Policy, dtype: str, n: int = 5, m: int = 5, n_enemies: int = 2):
        """
        :param player_policy: policy function that return player's action_index for given state
        :param dtype: optimization algorithm type RQL or DQN
        :param n: number of cells along x-axis, width
        :param m: number of cells along y-axis, height
        :param n_enemies: number of enemies
        """
        # evaluate action indexes
        self._action_index = {self._actions[index]: index for index in range(len(self._actions))}
        # set player's movement policy
        self._policy = player_policy.policy_function
        # set state type
        self._dtype = dtype
        # update world parameters in policy
        player_policy.set_world_properties(len(self._actions), n * m, self._dtype)
        # set x & y axis ranges
        self._rx = range(n)
        self._ry = range(m)
        # initiate n_actors at (0, 0) position
        self._init_actors(n_enemies)
        # reset actors to random positions
        self.reset()
        # save start transition in memory
        player_policy.push(self.transition())
        # set board object at size n, m and link actors to board
        self._board = CatchMeBoard(n, m, self._actors)

    def _init_actors(self, n: int) -> None:
        """
        initiate all actors: player & enemies at (0, 0) position
        :param n:  number of enemies, n >= 1
        """
        # initiate Player
        self._player = Player()
        # initiate enemies & prepare full actors list
        self._init_enemies(n)

    def _init_enemies(self, n: int) -> None:
        """
        initiate all enemies: enemies at (0, 0) position
        :param n:  number of enemies, n >= 1
        """
        # verify minimal number of actors is 1
        n = 1 if n < 1 else n
        # initiate enemies
        self._enemies = [Enemy() for _ in range(n)]
        # prepare full actors list
        self._actors = [self._player] + self._enemies

    def _state(self) -> State:
        """
        state definition: state is a board image, when every pixel is cell
        :return: world state
        """
        # prepare empty 'gray' board
        state = [[127 for _ in self._rx] for _ in self._ry]
        # set 'black' for player and 'white' for enemies
        for actor in self._actors:
            x, y = actor.curr_pos()
            state[y][x] = 255 if actor.type is 'enemy' else 0

        # return according to data type
        if self._dtype is 'DQN':
            return tensor(state, dtype=float, device=dev).unsqueeze(0)
        elif self._dtype is 'RQL':
            return tuple(reduce(lambda a, b: a + b, state))

    def _game_over(self) -> bool:
        """
        verify game status and update immediate reward value
        :return: true if game over
        """
        # if any enemy on same position with player the game is over
        game_over = any(map(lambda enemy: enemy.curr_pos() == self._player.curr_pos(), self._enemies))

        # update reward value
        if game_over:
            # reward for game over
            self._reward = -2
        else:
            # reward for making a move, while game not over
            self._reward = 1

        return game_over

    def _next_action(self) -> Action:
        """
        player's next action according to policy and current state
        :return: player's next action
        """
        return self._actions[self._policy(self._curr_state)]

    def draw(self):
        """draw world current state (board)"""
        sleep(1)                            # some delay to enable movement recognition
        self._board.draw()                  # draw board
        print('Steps:', self._step_count)   # print current step counter value

    def valid_pos(self, position: Position) -> bool:
        """
        validate that position is in the board
        :param position: (x, y) position index tuple
        :return: true if position in side the board
        """
        x, y = position
        return x in self._rx and y in self._ry

    def reset(self) -> None:
        """reset world parameters: clear step counter, set actors to random positions and evaluate initial transition"""
        # clear step counter
        self._step_count = 0

        # choose random player position
        self._player.reset(choice(self._rx), choice(self._ry))

        # choose random enemies positions, different from players position
        for enemy in self._enemies:
            enemy.reset(choice(self._rx), choice(self._ry))
            while self._player.curr_pos() == enemy.curr_pos():
                enemy.reset(choice(self._rx), choice(self._ry))

        # save initial full transition description
        self._prev_state = self._state()
        self._curr_state = self._prev_state
        self._last_action = (0, 0)
        self._reward = 0

    def dtype(self):
        return self._dtype

    def step_num(self) -> int:
        """
        get number of steps in current play
        :return: number of steps
        """
        return self._step_count

    def prev_state(self) -> State:
        """
        get world previous state
        :return: previous state
        """
        return self._prev_state

    def last_action(self) -> LastAction:
        """
        get player's last action
        :return: last action
        """
        # return according to data type
        if self._dtype is 'DQN':
            return tensor(self._action_index[self._last_action], device=dev)
        elif self._dtype is 'RQL':
            return self._action_index[self._last_action]

    def curr_state(self) -> State:
        """
        get world current state
        :return: current state
        """
        return self._curr_state

    def reward(self) -> Reward:
        """
        get immediate reward value
        :return: immediate reward
        """
        # return according to data type
        if self._dtype is 'DQN':
            return tensor(self._reward, dtype=float, device=dev)
        elif self._dtype is 'RQL':
            return self._reward

    def transition(self) -> Transition:
        """
        full transition description: previous state, last player's action, current state, immediate reward value
        :return: transition
        """
        return Transition(self.prev_state(), self.last_action(), self.curr_state(), self.reward())

    def play(self, silent: bool = False, smart_enemy: bool = True) -> bool:
        """
        perform single world step & report if game is not over
        :param silent: silent mode flag, if true: don't draw world state
        :param smart_enemy: smart mode flag, if false: enemies performs random moves
        :return: true if game not over
        """
        # player's step, depends on current enemies positions
        self._player.move(self._next_action(), self.valid_pos)

        # enemies step, depends on player new position in smart mode
        for enemy in self._enemies:
            if smart_enemy:
                # close gap to the player
                enemy.move_smart(self._player.curr_pos())
            else:
                # perform random move
                enemy.move(choice(self._actions), self.valid_pos)

        # save updated full transition description
        self._prev_state = self._curr_state
        self._last_action = self._player.last_action()
        self._curr_state = self._state()

        # update step counter
        self._step_count += 1

        # draw world state, if not in silent mode
        if not silent:
            self.draw()

        return not self._game_over()


if __name__ == '__main__':
    # todo: clear, finalize separate dqn & q-learning
    def training(max_epochs: int, print_num: int,
                 lr: float, gamma: float, eps_rand: float, eps_greedy: float, eps_min: float, alpha: float,
                 n_hidden: int, w_hidden: int,
                 memory_size: int, batch_size: int, random_batch: Optional[bool],
                 smart_enemy: bool,
                 algo_type: AlgoType,
                 net_path: str) -> (World, Policy, Dict[str, Plot]):
        """
        training routine
        :param max_epochs: maximal training epochs numbers
        :param print_num: print status every PRINT_NUM epochs
        :param lr: learning rate
        :param gamma: discount factor
        :param eps_rand: percent of exploration epochs at the beginning
        :param eps_greedy: percent of exploitation epochs at the end
        :param eps_min: minimal exploration percent, 0 == greedy
        :param alpha: smoothing constant (RMS prop)
        :param n_hidden: number of hidden layers
        :param w_hidden: hidden layer dimension
        :param memory_size: replay memory size
        :param batch_size: random batch size
        :param random_batch: use random batch or last game transitions for optimisation
        :param smart_enemy: use enemy smart policy if true, else use random policy
        :param algo_type: definition of optimization algorithm type RQL or DQN, reinforced-q-learning or deep-q-network
        :param net_path: network save path
        :return: world, policy, plots: world & player's action policy and dictionary of plots
        """

        # initiate graphs: number of steps per epoch, epsilon value per epoch, mean loss value per epoch
        _plot = {'steps': Plot(max_epochs, rolling={'method': 'mean', 'N': print_num}, figure_num=0),
                 'epsilon': Plot(max_epochs, title='Epsilon vs Epoch', ylabel='Epsilon', figure_num=1),
                 'loss': Plot(max_epochs, title='Loss vs Epoch', ylabel='Loss', figure_num=2,
                              rolling={'method': 'mean', 'N': print_num})}

        # initiate policy
        _policy = Policy(memory_size=memory_size, batch_size=batch_size, lr=lr, gamma=gamma, alpha=alpha,
                         epsilon=Epsilon(max_epochs=max_epochs, p_random=eps_rand, p_greedy=eps_greedy,
                                         explore_min=eps_min),
                         n_hidden=n_hidden, w_hidden=w_hidden)

        # initiate world
        _world = World(_policy, algo_type.item())

        # training
        tictoc = TicToc(max_epochs)  # start time counter
        for epoch in range(max_epochs):
            # perform world step until game is over, don't print board
            while _world.play(silent=True, smart_enemy=smart_enemy):
                _policy.push(_world.transition())  # save transition in memory
            _policy.push(_world.transition())  # save game over transition in memory

            # optimize policy
            if random_batch is None:
                optim_batch = batch_size if algo_type.item() is 'DQN' else _world.step_num()
            else:
                optim_batch = batch_size if random_batch else _world.step_num()

            _policy.optimize(batch_size=optim_batch, random_batch=random_batch)

            # update plots
            _plot['steps'].update(epoch, _world.step_num())
            _plot['epsilon'].update(epoch, _policy.eps())
            _plot['loss'].update(epoch, _policy.mean_loss())

            # print statistics every PRINT_NUM epochs
            if (epoch + 1) % print_num == 0:
                tictoc.toc()
                print('Epoch %7d;' % (epoch + 1), 'Step count: %5d;' % _plot['steps'].roll[epoch],
                      'loss: %7.3f; ' % _plot['loss'].roll[epoch], 'eta (s): %6.2f; ' % tictoc.eta(epoch))

            # update epsilon and reset players positions in world, if this is not last epoch
            if not epoch + 1 == max_epochs:
                _policy.epsilon_decay(epoch)        # update exploration/exploitation percent using decay algorithm
                _policy.target_net.load_state_dict(_policy.policy_net.state_dict())
                _policy.target_net.eval()
                _world.reset()                      # reset players in world
                _policy.push(_world.transition())   # save start transition in memory

        # save policy network
        _policy.save(net_path)

        # summarize training: maximal steps all over the training and training total time
        tictoc.toc()
        print('Max duration: %d;' % max(list(_plot['steps'].roll.values())[-len(_plot['steps'].roll) // 2:]),
              'Elapsed %.2f (s)' % tictoc.elapsed())

        return _world, _policy, _plot

    testing = False

    if testing:
        # Hyper params testing
        N = 100                                 # Number Hyper param sets
        p = list(arange(-6, 0, .01))            # list of powers
        ls = list(arange(0, 1, .01))            # list of linear range

        lr = [10**choice(p) for _ in range(N)]
        alpha = [.8 + 0.2 * choice(ls) for _ in range(N)]
        n_hidden_n = [choice(range(1, 10)) for _ in range(N)]
        w_hidden_n = [choice(range(25, 100)) for _ in range(N)]
        w_hidden_n.sort()
        memory_size_n = [choice(range(4, 9)) for _ in range(N)]
        batch_size_n = [choice(range(2, memory_size_n[n])) for n in range(N)]

        m = []
        for n in range(N):
            print('n = %d' % n)
            # hyper parameters & run training, todo: configure!!!
            world, policy, plots = training(max_epochs=50000,
                                            print_num=2000,
                                            lr=lr[n],
                                            gamma=0.999,
                                            eps_rand=0.1,
                                            eps_greedy=0.5,
                                            eps_min=0.,
                                            alpha=alpha[n],
                                            n_hidden=n_hidden_n[n],
                                            w_hidden=w_hidden_n[n],
                                            memory_size=memory_size_n[n],
                                            batch_size=batch_size_n[n],
                                            random_batch=True,
                                            smart_enemy=True,
                                            algo_type=AlgoType('DQN'),
                                            net_path='./mem/policy_net_%d.pkl' % n)
            m.append(max(filter(None, plots['steps'].roll.values())))
            print('n = %d, max steps = %.2f' % (n, m[n]))

        df = DataFrame(data={'lr': lr,
                             'alpha': alpha,
                             'n_hidden_n': n_hidden_n,
                             'w_hidden_n': w_hidden_n,
                             'memory_size_n': memory_size_n,
                             'batch_size_n': batch_size_n,
                             'm': m}, index=[n for n in range(N)])
        df.to_pickle('./result_tmp.pkl')
    else:
        world, policy, plots = training(max_epochs=1500000,
                                        print_num=2000,
                                        lr=0.000138,
                                        gamma=0.999,
                                        eps_rand=0.1,
                                        eps_greedy=0.7,
                                        eps_min=0.,
                                        alpha=0.974,
                                        n_hidden=3,
                                        w_hidden=95,
                                        memory_size=64,
                                        batch_size=64,
                                        random_batch=True,
                                        smart_enemy=True,
                                        algo_type=AlgoType('DQN'),
                                        net_path='./mem/policy_net_%d.pkl' % 2094)

        for key in plots:
            plots[key].plot()

        # play one game
        policy.set_greedy()  # set greedy policy
        world.reset()  # reset players in world
        world.draw()  # draw t=0
        # perform world step and print board until game is over
        while world.play(silent=False, smart_enemy=True):
            pass
