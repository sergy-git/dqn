from matplotlib import pyplot
from time import time
from statistics import mean


class Plot:
    def __init__(self, max_len, rolling=None, title='Training...', xlabel='Epoch', ylabel='Duration', figure_num = 0):
        self.line = {n: None for n in range(max_len)}
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._f_num = figure_num
        if rolling is not None:
            self.method = rolling['method']
            self.N = rolling['N']
            self.roll = {n: None for n in range(max_len)}
            self.sum = 0
        else:
            self.method = None
            self.roll = None

    def roll_append(self, idx):
        if self.method is 'mean':
            self.sum += self.line[idx]
            if idx + 1 >= self.N:
                self.roll[idx] = self.sum / self.N
                self.sum -= self.line[idx + 1 - self.N]

    def update(self, idx, value):
        self.line[idx] = value
        self.roll_append(idx)

    def plot(self):
        pyplot.figure(self._f_num)
        pyplot.clf()
        pyplot.title(self._title)
        pyplot.xlabel(self._xlabel)
        pyplot.ylabel(self._ylabel)
        pyplot.plot(list(self.line.values()))
        if self.roll is not None:
            pyplot.plot(list(self.roll.values()))
        pyplot.pause(0.00001)  # pause a bit so that plots are updated


class TicToc:
    def __init__(self, max_counts):
        self._max_counts = max_counts
        self._ts = time()
        self._elapsed = [0.]
        self._eta = []
        self._counts = [0]
        self._dt = []
        self._type = 'integral'

    def tic(self):
        self._ts = time()

    def toc(self):
        self._elapsed.append(time() - self._ts)

    def elapsed(self):
        return self._elapsed[-1]

    def eta(self, counts):

        self._counts.append(counts)
        self._dt.append((self._elapsed[-1] - self._elapsed[-2]) / (self._counts[-1] - self._counts[-2]))
        if self._type is 'integral' and len(self._eta) > 2 and mean(self._dt[-3:]) * 1.5 < self._dt[-1]:
            self._type = 'diff'

        if self._type is 'integral':
            self._eta.append(self._elapsed[-1] * (self._max_counts / (counts + 1) - 1))
        else:
            self._eta.append(self._dt[-1] * (self._max_counts - self._counts[-1]))

        return self._eta[-1]


def to_matrix(vector, lines_number):
    if len(vector) % lines_number == 0:
        return [vector[i:i + lines_number] for i in range(0, len(vector), lines_number)]
    else:
        return None


def draw_state(state, lines_number):
    rows = to_matrix(state, lines_number)
    for row in rows:
        print(row)


def conv2d_size_out(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1
# conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
# conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
