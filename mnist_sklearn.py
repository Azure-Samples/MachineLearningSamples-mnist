from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from azureml.core.run import Run
import os

data_path = '/tmp'

if "AZUREML_NATIVE_SHARE_DIRECTORY" in os.environ:
    print('use shared folder:')
    data_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
else:
    print('shared volume not enabled.')

print('data path:', data_path)
        
run = Run.get_submitted_run()

print('fetching MNIST data...')
mnist = fetch_mldata('MNIST original', data_home=data_path)

# use the full set with 70k records
#X, y = mnist['data'], mnist['target']

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



