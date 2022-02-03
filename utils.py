import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file,figure_num):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.figure(figure_num)
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plot_other_curve(scores, figure_file,figure_num,title = None):
    x = [i+1 for i in range(len(scores))]
    plt.figure(figure_num)
    plt.plot(x, scores)
    plt.title(title)
    plt.legend()
    plt.savefig(figure_file)
