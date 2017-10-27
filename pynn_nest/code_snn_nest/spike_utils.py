import numpy as np
import random
import pyNN.nest as p

def plot_spikes(spikes, title):
    if spikes is not None:
        fig, ax = plt.subplots()
        #plt.figure(figsize=(15, 5))
        ax.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        plt.title(title)

    else:
        print "No spikes received"
    plt.show()
