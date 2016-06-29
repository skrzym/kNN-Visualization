import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# Main File for kNN (k Nearest Neighbors) Algorithm
# From http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Idea is to do kNN on the cars dataset

# Handle Data:  Open the dataset from CSV and split into test/train datasets.
# Similarity:   Calculate the distance between two data instances.
# Neighbors:    Locate k most similar data instances.
# Response:     Generate a response from a set of data instances.
# Accuracy:     Summarize the accuracy of predictions.
# Main:         Tie it all together.


def load_data(filename, test_size=0.33):
    data = pd.read_csv(filename, header=None)
    train, test = train_test_split(data, test_size=test_size)
    train = train.sort_index(ascending=True)
    test = test.sort_index(ascending=True)
    return train, test


def euclidean_distance(element1, element2, cols):
    ssd = 0
    ssd += (element1[5] - element2[5])**2
    ssd += (element1[6] - element2[6])**2
    e_dist = math.sqrt(ssd)
    return e_dist


def euclidean_distance_orig(element1, element2, cols):
    ssd = 0
    for i in range(cols):
        ssd += (element1[i] - element2[i])**2
    e_dist = math.sqrt(ssd)
    return e_dist


def get_neighbors(training_set, test_element, k, original=False):
    distances = {}
    for i, r in training_set.iterrows():
        if original:
            distances[i] = (euclidean_distance_orig(list(r), test_element, 4))
        else:
            distances[i] = (euclidean_distance(list(r), test_element, 4))

    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
    index_list = []
    for i in range(k):
        index_list.append(sorted_distances[i][0])

    furthest_distance = sorted_distances[k][1]
    neighbors = training_set.loc[index_list]
    return neighbors, furthest_distance


def get_response(neighbors):
    tally = neighbors[4].value_counts()
    return tally.index.values[0]


def get_accuracy(pred, test):
    correct = 0.0
    for key in pred:
        if pred[key] == test[4].loc[key]:
            correct += 1
    return correct/len(test)


def main(file, k, test_size=0.33):
    train_set, test_set = load_data(filename=file, test_size=test_size)
    ####
    # Combine elements to condense factors from 4 -> 2 and allow 2 dimensional visualization
    train_set['x'] = train_set[0] + train_set[1]
    train_set['y'] = train_set[2] + train_set[3]
    test_set['x'] = test_set[0] + test_set[1]
    test_set['y'] = test_set[2] + test_set[3]
    ####

    '''
    virg_train_set = train_set[train_set[4] == 'Iris-virginica']
    seto_train_set = train_set[train_set[4] == 'Iris-setosa']
    vers_train_set = train_set[train_set[4] == 'Iris-versicolor']

    fig = plt.gcf()
    ax = plt.gca()
    # max_val = math.ceil(max([max(train_set[0]+train_set[1]), max(train_set[2]+train_set[3])]))
    # min_val = math.floor(min([min(train_set[0]+train_set[1]), min(train_set[2]+train_set[3])]))
    ax.set_xlim((5, 13))
    ax.set_ylim((1, 10))

    colors = {'Iris-virginica': 'b', 'Iris-setosa': 'c', 'Iris-versicolor': 'y', 'correct': 'g', 'wrong': 'r'}
    point_size = 300
    ax.scatter(virg_train_set['x'], virg_train_set['y'], s=point_size, c=colors['Iris-virginica'])
    ax.scatter(seto_train_set['x'], seto_train_set['y'], s=point_size, c=colors['Iris-setosa'])
    ax.scatter(vers_train_set['x'], vers_train_set['y'], s=point_size, c=colors['Iris-versicolor'])
    '''
    predictions = {}
    for i, r in test_set.iterrows():
        current_neighbors, furthest_distance = get_neighbors(train_set, list(r), k, True)
        current_response = get_response(current_neighbors)
        predictions[i] = current_response
        # print "> predicted=", repr(current_response), "actual=", repr(r[4])
        '''
        if current_response == r[4]:
            ax.scatter(r['x'], r['y'], s=point_size, c=colors['correct'], edgecolor=colors['correct'])
            ax.scatter(r['x'], r['y'], s=point_size/2, c=colors[current_response], edgecolor=colors['correct'])
            #circle = plt.Circle((r['x'], r['y']), furthest_distance, fill=False, edgecolor=colors['correct'])
            #ax.add_artist(circle)
        else:
            ax.scatter(r['x'], r['y'], s=point_size, c=colors['wrong'], edgecolor=colors['wrong'])
            ax.scatter(r['x'], r['y'], s=point_size/2, c=colors[current_response], edgecolor=colors[current_response])
            circle = plt.Circle((r['x'], r['y']), furthest_distance, fill=False, edgecolor=colors['wrong'])
            ax.add_artist(circle)
        '''
    # print get_accuracy(predictions, test_set)

    #plt.show()

    return get_accuracy(predictions, test_set)


def main_test(file, k, test_size=0.33):
    train_set, test_set = load_data(filename=file, test_size=test_size)
    ####
    # Combine elements to condense factors from 4 -> 2 and allow 2 dimensional visualization
    train_set['x'] = train_set[0] + train_set[1]
    train_set['y'] = train_set[2] + train_set[3]
    test_set['x'] = test_set[0] + test_set[1]
    test_set['y'] = test_set[2] + test_set[3]
    ####

    predictions1 = {}
    predictions2 = {}
    for i, r in test_set.iterrows():
        current_neighbors1, furthest_distance1 = get_neighbors(train_set, list(r), k, False)
        current_response1 = get_response(current_neighbors1)
        predictions1[i] = current_response1

        current_neighbors2, furthest_distance2 = get_neighbors(train_set, list(r), k, True)
        current_response2 = get_response(current_neighbors2)
        predictions2[i] = current_response2

    print get_accuracy(predictions1, test_set)
    print get_accuracy(predictions2, test_set)


print main_test('iris.data.csv', 5, 0.33)

'''
# MULTI SIM TESTING
acc_list = []
test_list = []
min_k = 2
max_k = 7
tests_per_k = 100
for k in range(min_k, max_k + 1):
    for test in range(tests_per_k):
        test_list.append(main('iris.data.csv', k, 0.33))
        print 'K:' + str(k) + '  Test:' + str(test) + '  Acc:' + str(test_list[-1]*100) + '%'
    acc_list.append(np.mean(test_list))
fig = plt.gcf()
ax = plt.gca()
ax.set_xlim((0, max_k + 1))
ax.set_ylim((0.80, 1.0))
plt.plot(range(min_k, max_k + 1), acc_list)
plt.show()
'''

"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.gcf()
ax = plt.gca()
ax.set_xlim((5, 13))
ax.set_ylim((1, 10))


# initialization function: plot the background of each frame
def init():
    pass


# animation function.  This is called sequentially
def animate(i):
    return ax

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

