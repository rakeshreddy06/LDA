import argparse
import numpy as np  

def generate_random_data(n_samples_per_class=50, n_features=4, n_classes=3, seed=20):
    np.random.seed(20)
    X = []
    y = []
    for class_label in range(n_classes):
        class_data = np.random.randn(n_samples_per_class, n_features) + class_label * 2 
        X.append(class_data)
        y.extend([class_label] * n_samples_per_class)

    X = np.vstack(X)
    y = np.array(y)
    
    return X, y

def write_data(output_file, X, y):
    with open(output_file, 'w') as file:
        for i in range(len(X)):
            line = ','.join(map(str, X[i])) + ',' + str(y[i]) + '\n'
            file.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type = int, help="Number of samples")
    parser.add_argument("-f", type = int, help="Number of features")
    parser.add_argument("-c", type = int, help="Number of classes")
    parser.add_argument("-seed", type=int, help="A seed to control randomnes")
    parser.add_argument("-output_file", type=str, help="Path to output file")
    args = parser.parse_args()
    X, y = generate_random_data(args.N, args.f, args.c, args.seed)
    write_data(args.output_file, X,y)

if __name__=="__main__":
    main()

