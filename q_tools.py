from matplotlib import pyplot as plt


class Plot:
    def __init__(self):
        self.line = []

    def update(self, value, silent=False):
        self.line.append(value)
        if not silent:
            self.plot()

    def plot(self):
        plt.figure(0)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Epoch')
        plt.ylabel('Duration')
        plt.plot(self.line)
        plt.pause(0.00001)  # pause a bit so that plots are updated
