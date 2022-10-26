import numpy
from sklearn import neural_network
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
    sets, labels = numpy.zeros([file_count, 1024], int), numpy.zeros([file_count, 10], int)
    for i in range(file_count):
        labels[i][int(file_list[i].split('_')[0])] = 1.0
        sets[i] = imgtomatrix('images/' + path + '/' + file_list[i])
    return sets, labels


def run_mlp(hidden, rate, iter, solver):

    # read training data and split into testing and training sets
    training_sets, training_labels = read_image_data('train_digit_images')
    
    # initialize the MLPClassfiere model
    mlp = neural_network.MLPClassifier( hidden_layer_sizes=(hidden,), 
                                        activation='logistic', 
                                        solver=solver,
                                        learning_rate_init=rate, 
                                        max_iter=iter)
   
    # fit the model
    mlp.fit(training_sets, training_labels)

    # read test data
    sets, labels = read_image_data('test_digit_images')

    # predict from test data
    results = mlp.predict(sets)

    incorrect_preds = 0

    # check for number of incorrectly predicted labels
    for i in range(len(sets)):
        if numpy.sum(results[i] == labels[i]) < 10:
            incorrect_preds += 1

    
    print("Hidden Layer Nodes", hidden)
    print("Learning Rate     ", rate)
    print("Total             ", + len(sets) )
    print("No. of Errors     ", incorrect_preds) 
    print("Error %           ", incorrect_preds / float(len(sets)))
    print("Accuracy %        ", 1 - incorrect_preds / float(len(sets)))
  
    return float(1 - incorrect_preds / float(len(sets)))


# test the model with different learning rates
results = []
for lr in [0.1, 0.01, 0.001, 0.0001]:
    acc = run_mlp(100, lr, 100000, 'sgd')
    results.append((lr, acc))

# once each model is tested, plot the results for each learning rate
plt.plot([x[0] for x in results], [x[1] for x in results])
plt.xlabel('lr')
plt.ylabel('accuracy')
plt.title('mlp graph')
plt.show()