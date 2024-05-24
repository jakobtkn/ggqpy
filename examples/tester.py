import numpy as np
import dill as pickle

class Tester():
    name = None


    def __init__(self, method, name = None):
        self.num_nodes = list()
        self.z = list()
        self.method = method
        self.abs_error = None
        self.rel_error = None

        if name == None:
            self.name = str(method)
        else:
            self.name = name
    
    def perform_test(self, *args):
        value, nodes = self.method(* args)
        self.num_nodes.append(nodes)
        self.z.append(value)

    def compute_error(self, target: float):
        self.abs_error = abs(np.array(self.z) - target)
        self.rel_error = abs((np.array(self.z) - target)/target)

    def save(self, folder):
        with open(f"data/{folder}/{self.name}.pkl", "wb") as output:
            pickle.dump(self, output)

def load_tester(folder: str, name: str):
    filename = f"data/{folder}/{name}.pkl"
    with open(filename, 'rb') as input_:
        tester = pickle.load(input_)

    return tester