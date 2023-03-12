import pickle
import random



def load_data():
    train_fraction = 0.9

    with open('x.pkl', 'rb') as f:
        x = pickle.load(f)

    with open('y.pkl', 'rb') as f:
        y = pickle.load(f)

    xy = list(zip(x,y))
    random.shuffle(xy)
    x, y = zip(*xy)

    n = int(train_fraction * len(x))
    x_train, y_train = x[:n], y[:n]
    x_val, y_val = x[n:], y[n:]

    print("train:", len(x_train))
    print("val:", len(x_val))
    return x_train, y_train, x_val, y_val

def pad_elements(elements, target_size):
    pad_token = 16
    for e in elements:
        while len(e) < target_size:
            e.append(pad_token)
    return elements