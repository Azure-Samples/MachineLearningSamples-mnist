from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from azureml.core.run import Run
from PlotUtils import plot_digits

data_path = '/tmp'
os.makedirs('./outputs', exist_ok=True)
    
if "AZUREML_NATIVE_SHARE_DIRECTORY" in os.environ:
    print('use shared folder:')
    data_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
else:
    print('shared volume not enabled.')

print('data path:', data_path)
        
run = Run.get_submitted_run()

print('fetching MNIST data...')
mnist = fetch_mldata('MNIST original', data_home=data_path)

# use the full set with 70,000 records
X, y = mnist['data'], mnist['target']

print('plot some digits...')
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()
plt.savefig('./outputs/digits.png', format='png', dpi=300)

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



