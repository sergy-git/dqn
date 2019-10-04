from matplotlib import pyplot
from statistics import mean


class Plot:
    def __init__(self, rolling=None):
        self.line = []
        if rolling is not None:
            self.method = rolling['method']
            self.N = rolling['N']
            self.roll = []
        else:
            self.method = None
            self.roll = None

    def roll_append(self):
        if self.method is 'mean':
            self.roll.append(None if len(self.line) < self.N else mean(self.line[-self.N:]))

    def update(self, value, silent=False):
        self.line.append(value)
        self.roll_append()
        if not silent:
            self.plot()

    def plot(self):
        pyplot.figure(0)
        pyplot.clf()
        pyplot.title('Training...')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Duration')
        pyplot.plot(self.line)
        if self.roll is not None:
            pyplot.plot(self.roll)
        pyplot.pause(0.00001)  # pause a bit so that plots are updated
