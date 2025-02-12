import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# for data visualization 
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
import numpy as np


def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')


def comparison_plot(metrics, classes, figsize=(10, 6), smooth_test=False):
    if smooth_test:
        for cls in classes:
            cls['test acc'] = smooth(cls['test acc'])

    for metric in metrics:
        plt.figure(figsize=figsize)
        for cls in classes:
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.plot(cls[metric], label=cls['name'])
        plt.title(f"{metric}")
        plt.legend()  
        plt.show()


