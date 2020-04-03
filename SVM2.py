# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from MLII.mnist_helpers import show_some_digits
#fetch  mnist dataset
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

digits= datasets.load_digits()
images = digits.data[:]
targets = digits.target[:]
print(images.shape[1])
len(images)
print(targets.shape[2])
len(targets)

# Let's have a look at the random 16 images,
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array
#pick  random indexes from 0 to size of our dataset

rand_idx = np.random.choice(images.shape[0], 24)
print(rand_idx)
images[rand_idx]
targets[rand_idx]
images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))
title_text='Digit {}'
img = plt.figure(1, figsize=(15, 12), dpi=160)
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(np.ceil(24 / 6.0), 6, index + 1)
    plt.axis('off')
    # each image is flat, we have to reshape to 2D array 28x28-784
    plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(title_text.format(label))
plt.show()


#show_some_digits(images, digits)

#---------------- classification begins -----------------

#full dataset classification
X_data = images/255.0
len(X_data)
print(X_data)
Y = targets

#split data to train and test
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

clf = svm.SVC(C=5, gamma=.05, kernel='rbf')
clf.fit(X_train,y_train)

ans = clf.predict(X_test)
confusion_matrix(y_test,ans)
accuracy_score(y_test,ans)
