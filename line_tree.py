from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
leafs = 0

def find_split(points,lr=1e-1,tol=0.05):
    N = len(points)
    if np.std(points["f"],ddof=0.5) < tol*np.exp(-N/100):
        return None
    x=np.stack((points["x"],points["y"]))
    x_n = x-np.mean(x,axis=1,keepdims=True)
    x_n /= np.std(x,axis=1,keepdims=True)+1e-5
    slope = np.random.randn(1,2)
    bias = 0#np.random.randn(1)
    #Loss is now L=sum((f-tanh(w@x+b))**2)
    f = points["f"]-np.mean(points["f"])
    f /= np.std(f)
    #print(np.sum((f-np.tanh(slope@x+bias))**2))
    for k in range(500):
        f_hat = np.tanh(slope@x_n+bias)
        residual = f-f_hat
        Dw = np.mean(-residual*(1-f_hat**2)*x_n,axis=1)
        Db = np.mean(-residual*(1-f_hat**2))
        bias -= lr*Db
        slope -= lr*Dw
        if abs(Db) < 1e-3 and np.all(np.abs(Dw)<1e-3):
            break
    slope = slope/(np.std(x,axis=1)+1e-5)
    bias = bias-slope@np.mean(x,axis=1)
    #print(np.sum((f-np.tanh(slope@x+bias))**2))
    return (slope,bias)


def tree_recursion(points):
    global leafs
    weights = find_split(points)
    if weights is None:
        leafs += 1 
        return np.mean(points["f"])
    slope,bias = weights
    x=np.stack((points["x"],points["y"]))
    left_idx = (slope@x+bias<0).squeeze()
    right_idx = np.invert(left_idx)
    if not np.any(left_idx) or not np.any(right_idx):
        #print("hej")
        leafs += 1 
        return np.mean(points["f"])
    left_pts, right_pts = points[left_idx],points[right_idx]
    
    left_splits = tree_recursion(left_pts)
    right_splits = tree_recursion(right_pts)
    return (weights,(left_splits,right_splits))

def find_value(pt,tree):
    if type(tree) is not tuple:
        return tree
    slope,bias = tree[0]
    left_path,right_path = tree[1][0],tree[1][1]
    go_left = slope@pt+bias<0
    if go_left:
        return find_value(pt,left_path)
    else:
        return find_value(pt,right_path)

image = np.asarray(Image.open("cameraman.jpg"))
image = np.mean(image,axis=2)
image = image[::4,::4]/255.0
points = []
size = len(image)
for x in range(size):
    for y in range(size):
        points.append((x/size,y/size,image[y,x]))
dtype = [("x",np.float),("y",np.float),("f",np.float)]
points = np.array(points,dtype=dtype)
tree = tree_recursion(points)
scale=1
image = np.zeros((scale*size,scale*size))
for x in range(scale*size):
    for y in range(scale*size):
        image[y,x] = find_value(np.array([x/size/scale,y/size/scale]),tree)
print(leafs)
plt.imshow(image,cmap="gray")
plt.show()
