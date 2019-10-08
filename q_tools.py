from matplotlib import pyplot


class Plot:
    def __init__(self, max_len, rolling=None):
        self.line = {n: None for n in range(max_len)}
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
        pyplot.figure(0)
        pyplot.clf()
        pyplot.title('Training...')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Duration')
        pyplot.plot(list(self.line.values()))
        if self.roll is not None:
            pyplot.plot(list(self.roll.values()))
        pyplot.pause(0.00001)  # pause a bit so that plots are updated
