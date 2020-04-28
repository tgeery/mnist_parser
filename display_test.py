import os, matplotlib.pyplot as plt, numpy as np

if not os.path.exists("t10k-images-idx3-ubyte"):
    os.system("wget yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    os.system("gunzip t10k-images-idx3-ubyte.gz")

if not os.path.exists("t10k-labels-idx1-ubyte"):
    os.system("wget yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    os.system("gunzip t10k-labels-idx1-ubyte.gz")

with open("t10k-images-idx3-ubyte", "rb") as f_images:
    with open("t10k-labels-idx1-ubyte", "rb") as f_labels:
        magic_image = f_images.read(4)
        magic_label = f_labels.read(4)
        num_images = int.from_bytes(f_images.read(4), byteorder='big')
        num_rows = int.from_bytes(f_images.read(4), byteorder='big')
        num_cols = int.from_bytes(f_images.read(4), byteorder='big')
        num_items = int.from_bytes(f_labels.read(4), byteorder='big')
        print("magic labels {}, images: {}".format(magic_label, magic_image))
        print("number of items {}, images {}".format(num_items, num_images))
        print("number of rows {}".format(num_rows))
        print("number of columns {}".format(num_cols))
        for i in range(num_images):
            lbl = f_labels.read(1)
            print("label: {}".format(lbl))
            arr = np.frombuffer(f_images.read(num_rows * num_cols), dtype='uint8')
            arr = arr.reshape((num_rows, num_cols))
            plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
            plt.show()

