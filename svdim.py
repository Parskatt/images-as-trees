from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


image = np.asarray(Image.open("mamma.jpg"))
image = np.mean(image,axis=2)
image = image[::2,::2]/255.0

U,S,VT = np.linalg.svd(image)
plt.plot(S)
plt.show()
S = np.diag(S)
k=20
plt.imshow(U[:,:k]@S[:k,:]@VT,cmap="gray")
plt.show()
print("wow")