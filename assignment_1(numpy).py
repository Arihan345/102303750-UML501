# -*- coding: utf-8 -*-
"""Assignment-1(NumPy).ipynb

import numpy as np
np1=np.array([1,2,3,4,5,6])
np1=np1[::-1]
print(np1)

import numpy as np
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
a1=array1.flatten()
a2=array1.ravel()
print(a1)
print(a2)

import numpy as np
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print(arr1 == arr2)
print(np.array_equal(arr1, arr2))

import numpy as np
x = np.array([1,2,3,4,5,1,2,1,1,1])
counts = np.bincount(x)
most_frequent_value = np.argmax(counts)
indices = np.where(x == most_frequent_value)
print(most_frequent_value)
print(indices)

import numpy as np
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
counts = np.bincount(y)
most_frequent_value = np.argmax(counts)
indices = np.where(y == most_frequent_value)
print(most_frequent_value)
print(indices)

import numpy as np
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print(np.sum(gfg))
print(np.sum(gfg, axis=1))
print(np.sum(gfg, axis=0))

import numpy as np
from numpy.linalg import eig, inv, det

n_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]])
print(np.trace(n_array))
eigen_values, eigen_vectors = eig(n_array)
print(eigen_values)
print(eigen_vectors)
print(inv(n_array))
print(det(n_array))

import numpy as np
p = np.array([[1, 2], [2, 3]])
q = np.array([[4, 5], [6, 7]])
matrix_product = np.dot(p, q)
print(matrix_product)
covariance = np.cov(p, q)
print(covariance)

import numpy as np
q = np.array([[4, 5, 1], [6, 7, 2]])
p = np.array([[1, 2], [2, 3], [4, 5]])
matrix_product = np.dot(q, p)
print(matrix_product)
covariance = np.cov(q, p)
print(covariance)

import numpy as np
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])
inner_product = np.inner(x, y)
print(inner_product)
outer_product = np.outer(x, y)
print(outer_product)
cartesian_product = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)
print(cartesian_product)

import numpy as np
array = np.array([[1, 2, 3],[-4, 5, -6]])
absolute_values = np.abs(array)
print(absolute_values)

import numpy as np
array = np.array([[1, 2, 3],[-4, 5, -6]])
print(np.percentile(array, [25, 50, 75]))
print(np.percentile(array, [25, 50, 75], axis=0))
print(np.percentile(array, [25, 50, 75], axis=1))

import numpy as np
array = np.array([[1, 2, 3],[-4, 5, -6]])
print(np.mean(array))
print(np.median(array))
print(np.std(array))
print(np.mean(array, axis=0))
print(np.median(array, axis=0))
print(np.std(array, axis=0))
print(np.mean(array, axis=1))
print(np.median(array, axis=1))
print(np.std(array, axis=1))

import numpy as np
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print(a)
print(np.floor(a))
print(np.ceil(a))
print(np.trunc(a))
print(np.round(a))

import numpy as np
array = np.array([10, 52, 62, 16, 16, 54, 453])
sorted_array = np.sort(array)
print(sorted_array)

import numpy as np
array = np.array([10, 52, 62, 16, 16, 54, 453])
sorted_indices = np.argsort(array)
print(sorted_indices)

import numpy as np
array = np.array([10, 52, 62, 16, 16, 54, 453])
sorted_array = np.sort(array)
smallest_4 = sorted_array[:4]
print(smallest_4)

import numpy as np
array = np.array([10, 52, 62, 16, 16, 54, 453])
sorted_array = np.sort(array)
largest_5 = sorted_array[-5:]
print(largest_5)

import numpy as np
array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
integer_elements = array[array == array.astype(int)]
print(integer_elements)
float_elements = array[array != array.astype(int)]
print(float_elements)

import numpy as np
from PIL import Image

def img_to_array(path):
    try:
        img = Image.open(path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            print("Grayscale image detected.")
            file_name = "grayscale_image.txt"
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            print("RGB image detected.")
            file_name = "rgb_image.txt"
        else:
            print("Image format not supported.")
            return

        np.savetxt(file_name, img_array.reshape(-1, img_array.shape[-1] if len(img_array.shape) == 3 else 1), fmt='%d')
        print(file_name)
    except FileNotFoundError:
        print(f" not found")

import numpy as np
loaded_array = np.loadtxt("grayscale_image.txt").reshape(100, 100)
print("File loaded successfully.")
