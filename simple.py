from q_tools import Plot


MAX_EPOCHS = 100000
p = Plot(MAX_EPOCHS, title='Epsilon vs Epoch', ylabel='Epsilon')

_random = .1
_max_epochs = round(MAX_EPOCHS * (1 - _random))
for epoch in range(MAX_EPOCHS - _max_epochs):
    eps = 1
    p.update(epoch, eps)
for epoch in range(_max_epochs):
    eps = (1 - epoch ** 2 / _max_epochs ** 2)*.90 + 0.10
    p.update(epoch + MAX_EPOCHS - _max_epochs, eps)

p.plot()
