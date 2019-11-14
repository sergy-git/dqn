from matplotlib import pyplot
from pandas import read_pickle

file = './result_191114.pkl'
df = read_pickle(file)

pyplot.plot(df.m)
