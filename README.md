# Azure - Machine Learning Samples - MNIST

### Creating and Executing sample
1) Create an Azure ML Project using this sample as a template.
    ```sh
    $ az ml project create -n mnist --path /tmp -w my_wg -g my_rg --template-url https://github.com/Azure/MachineLearningSamples-mnist
    ```
2) Change in the "minst" directory.
    ```sh
	$ cd mnist
    ```
3) Execute the script.
    ```sh
    $ az ml experiment submit -c local ./mnist_tf.py
    ```


