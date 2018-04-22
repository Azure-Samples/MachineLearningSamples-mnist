import matplotlib
matplotlib.use('agg')
import  matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(mx, save_file_name='mx.png'):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(mx, cmap=plt.cm.bone)
    ticks = np.arange(0, 10, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    fig.colorbar(cax)
    plt.savefig(save_file_name, format='png', dpi=300)

def plot_digits(instances, images_per_row=10, save_file_name='digits.png', **options):
    plt.figure(figsize=(9,9))
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
    plt.savefig(save_file_name, format='png', dpi=300)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
