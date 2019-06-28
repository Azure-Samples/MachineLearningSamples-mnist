> **NOTE** This content is no longer maintained. Visit the [Azure Machine Learning Notebook](https://github.com/Azure/MachineLearningNotebooks) project for sample Jupyter notebooks for ML and deep learning with Azure Machine Learning.

```shell

# create an Azure ML project using this sample as a template
$ az ml project create -n mnist --path /tmp -w my_wg -g my_rg --template-url https://github.com/Azure/MachineLearningSamples-mnist

# change directory
$ cd mnist

# run a script
$ az ml experiment submit -c local ./mnist_tf.py

```
