from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import numpy as np
import os

from azureml.core.run import Run
from PlotUtils import plot_digits, plot_confusion_matrix

data_path = '/tmp'
os.makedirs('./outputs', exist_ok=True)
    
if "AZUREML_NATIVE_SHARE_DIRECTORY" in os.environ:
    print('use shared folder.')
    data_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
else:
    print('shared volume not enabled.')

print('data path:', data_path)
        
run = Run.get_submitted_run()

print('fetching MNIST data...')
mnist = fetch_mldata('MNIST original', data_home=data_path)

# use the full set with 70,000 records
X, y = mnist['data'], mnist['target']

print('plotting some digits...')
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(instances=example_images, images_per_row=10, save_file_name='./outputs/digits.png')

# use a random subset of n records to reduce training time.
n = 5000
shuffle_index = np.random.permutation(70000)[:n]
X, y = mnist['data'][shuffle_index], mnist['target'][shuffle_index]

print('X: ', X.shape)
print('y: ', y.shape)
print('labels: ', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

lr = LogisticRegression()
print("training a logistic regression model...")
lr.fit(X_train, y_train)
print(lr)

y_hat = lr.predict(X_test)
acc = np.average(np.int32(y_hat == y_test))
run.log('accuracy', acc)

print('Overall accuracy:', acc)

conf_mx = confusion_matrix(y_test, y_hat)
print('Confusion matrix:')
print(conf_mx)

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

print('plotting a normalized confusionn matrix...')
plot_confusion_matrix(norm_conf_mx, save_file_name='./outputs/mx.png')


