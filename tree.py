from PIL import Image
import numpy as np

leafs = 0

def find_split(points,tol=10):
    N = len(points)
    start_cost = N*np.std(points["f"],ddof=0.99)
    best_cost = start_cost
    x_sorted_pts = np.sort(points,order="x")
    split = None
    for idx in range(1,N):
        cost = idx*np.std(x_sorted_pts[:idx]["f"],ddof=0.99)+(N-idx)*np.std(x_sorted_pts[idx:]["f"],ddof=0.99)
        if cost < best_cost:
            best_cost = cost
            split = ("x",idx)
    y_sorted_pts = np.sort(points,order="y")
    for idx in range(1,N):
        cost = idx*np.std(y_sorted_pts[:idx]["f"],ddof=0.99)+(N-idx)*np.std(y_sorted_pts[idx:]["f"],ddof=0.99)
        if cost < best_cost:
            best_cost = cost
            split = ("y",idx)
    if best_cost > start_cost-tol:
        return None
    return split


def tree_recursion(points):
    split = find_split(points)
    if split is None:
        global leafs
        leafs += 1 
        return np.mean(points["f"])
    variable,idx = split
    points = np.sort(points,order=variable)
    left_pts, right_pts = points[:idx],points[idx:]
    left_splits = tree_recursion(left_pts)
    right_splits = tree_recursion(right_pts)
    return ((variable,points[idx][variable]),(left_splits,right_splits))

def find_value(pt,tree):
    if type(tree) is not tuple:
        return tree
    decision = tree[0]
    left_path,right_path = tree[1][0],tree[1][1]
    go_left = pt[decision[0]]<=decision[1]
    if go_left:
        return find_value(pt,left_path)
    else:
        return find_value(pt,right_path)
image = np.asarray(Image.open("cameraman.jpg"))
image = np.mean(image,axis=2)
image = image[::4,::4]
#image = np.zeros([100,100])
#image[50:] = 1
points = []
for x in range(64):
    for y in range(64):
        points.append((x,y,image[y,x]))
dtype = [("x",int),("y",int),("f",np.float)]
points = np.array(points,dtype=dtype)
tree = tree_recursion(points)
for x in range(64):
    for y in range(64):
        image[y,x] = find_value({"x":x,"y":y},tree)
print(leafs)
image = Image.fromarray(image)
image=image.convert('RGB')
image.show()#(open("boxes.jpg","w"))
#print(nice)