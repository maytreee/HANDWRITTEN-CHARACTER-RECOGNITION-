import numpy
from sklearn import neighbors
import matplotlib.pyplot as plt
import imageio
from os import listdir

def imgtomatrix(f):
    image = imageio.imread(f)
    res = numpy.zeros([1024], int)
    x, y = image.shape[0], image.shape[1]
    for xx in range(x):
        for yy in range(y):
            res[xx * 32 + yy] = 1  
            if (list(image[xx][yy]) == [255, 255, 255]):
                res[xx * 32 + yy] = 0
    return res


def read_image_data(path):
    file_list = listdir('images/' + path)
    file_count = len(file_list)
    sets, labels = numpy.zeros([file_count, 1024], int),  numpy.zeros([file_count])
    for i in range(file_count):
        labels[i] = int(file_list[i].split('_')[0])
        sets[i] = imgtomatrix('images/' + path + '/' + file_list[i])
    return sets, labels


def run_knn(k):

    # read training data and split into testing and training sets
    training_sets, training_labels = read_image_data('train_digit_images')

    # initialize the KNeighborsClassifier model
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)

    # fit the model
    knn.fit(training_sets, training_labels)

    # read test data
    sets, labels = read_image_data('test_digit_images')

    # predict from test data
    results = knn.predict(sets)

    # check for number of incorrectly predicted labels
    incorrect_preds = numpy.sum(results != labels)

    print("Neighbor nodes", + k)
    print("Total         ", + len(sets) )
    print("No. of Errors ", incorrect_preds) 
    print("Error %       ", incorrect_preds / float(len(sets)))
    print("Accuracy %    ", 1 - incorrect_preds / float(len(sets)))

    return float(1 - incorrect_preds / float(len(sets))), knn

# test the model with different values of k
results = []
for k in [3, 5, 7, 9, 11]:
    acc, _ = run_knn(k)
    results.append((k, acc))

# once each model is tested, plot the results for each k
plt.plot([x[0] for x in results], [x[1] for x in results])
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('knn graph')
plt.show()

