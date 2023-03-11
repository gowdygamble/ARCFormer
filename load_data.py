import pickle
import random



def load_data():
    with open('x.pkl', 'rb') as f:
        x = pickle.load(f)

    with open('y.pkl', 'rb') as f:
        y = pickle.load(f)

    xy = list(zip(x,y))
    random.shuffle(xy)
    x, y = zip(*xy)

    print("data length:", len(x), len(y))
    return x, y

