from matplotlib import pyplot as plt


def curve(xs, ys):
    plt.plot(xs, ys, 'o')
    plt.plot(xs, ys, '#1f77b4')
    plt.grid()
