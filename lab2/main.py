from kohonen import KohonenSOM
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def normalize(data):
    """Min-max normalization"""
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)


def generate_class_samples(mean, std, n):
    """Generate samples for a class"""
    return np.random.normal(mean, std, (n, len(mean)))

def shuffle_data(data, labels):
    indices = np.random.permutation(len(data))
    return data[indices], np.array(labels)[indices]

def generate_dataset(train_n, test_n):
    """
    Generates training and test data for ship classification.
    15 properties per ship.
    """

    classes = {
        "cargo_ship": {
            "mean": [220, 32, 12, 70000, 24, 18, 50000, 20, 25, 5000, 30000, 80, 45, 8, 4],
            "std":  [15, 3, 2, 6000, 2, 2, 5000, 5, 3, 400, 3000, 5, 4, 1, 1],
        },
        "oil_tanker": {
            "mean": [300, 50, 20, 150000, 20, 15, 120000, 15, 30, 9000, 35000, 70, 60, 9, 3],
            "std":  [20, 4, 3, 12000, 2, 2, 9000, 4, 4, 600, 4000, 6, 6, 1, 1],
        },
        "passenger_ferry": {
            "mean": [180, 28, 7, 35000, 28, 22, 5000, 2000, 120, 3000, 25000, 90, 10, 6, 8],
            "std":  [10, 2, 1, 4000, 2, 2, 1000, 200, 10, 300, 2000, 6, 2, 1, 1],
        },
        "destroyer": {
            "mean": [155, 20, 6, 9000, 35, 28, 2000, 350, 320, 2000, 60000, 200, 30, 9, 9],
            "std":  [8, 1.5, 1, 1000, 3, 2, 400, 50, 30, 200, 5000, 15, 4, 1, 1],
        },
        "fishing_vessel": {
            "mean": [45, 10, 4, 900, 16, 12, 100, 10, 12, 200, 1500, 30, 15, 5, 7],
            "std":  [5, 1, 0.5, 150, 2, 2, 40, 4, 2, 40, 200, 5, 2, 1, 1],
        },
    }

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for label, params in classes.items():
        train_samples = generate_class_samples(params["mean"], params["std"], train_n)
        test_samples = generate_class_samples(params["mean"], params["std"], test_n)

        train_data.append(train_samples)
        test_data.append(test_samples)

        train_labels += [label] * train_n
        test_labels += [label] * test_n

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    # normalize together so scaling is consistent
    all_data = np.vstack((train_data, test_data))
    all_data = normalize(all_data)

    train_data = all_data[:len(train_data)]
    test_data = all_data[len(train_data):]

    train_data, train_labels = shuffle_data(train_data, train_labels)
    test_data, test_labels = shuffle_data(test_data, test_labels)

    return train_data, train_labels, test_data, test_labels

def label_som(som, train_data, train_labels):
    neuron_classes = defaultdict(list)

    # map training samples to neurons
    for x, label in zip(train_data, train_labels):
        coord = som.map_vector(x)
        neuron_classes[coord].append(label)

    # assign majority class
    neuron_labels = {}

    for neuron, labels in neuron_classes.items():
        neuron_labels[neuron] = Counter(labels).most_common(1)[0][0]

    return neuron_labels
    
def predict(som, neuron_labels, x):
    coord = som.map_vector(x)

    if coord in neuron_labels:
        return neuron_labels[coord]
    else:
        return "unknown"

def analyze_effectiveness_from_lr_rectangular(train_data, train_labels, test_data, test_labels, learning_rates):
    results = []

    for lr in learning_rates:
        som = KohonenSOM(grid_size=(10,10), input_dim=15, learning_rate=lr)
        som.train_rectangular(train_data)
        neuron_labels = label_som(som, train_data, train_labels)
        error = classification_error(som, neuron_labels, test_data, test_labels)
        results.append(error)

        print(f"lr={lr} error={error}")
    
    return results

def analyze_effectiveness_from_size_rectangular(train_ns, test_n):
    results = []

    for train_n in train_ns:
        train_data, train_labels, test_data, test_labels = generate_dataset(train_n, test_n)
        som = KohonenSOM(grid_size=(10,10), input_dim=15)
        som.train_rectangular(train_data)
        neuron_labels = label_som(som, train_data, train_labels)
        error = classification_error(som, neuron_labels, test_data, test_labels)
        results.append(error)

        print(f"train_n={train_n} error={error}")
        
    return results    

def classification_error(som, neuron_labels, test_data, test_labels):
    wrong = 0

    for x, true_label in zip(test_data, test_labels):
        pred = predict(som, neuron_labels, x)
        if pred != true_label:
            wrong += 1

    return wrong / len(test_labels)

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = generate_dataset(20, 2)

    print("Training samples:", train_data.shape)
    print("Test samples:", test_data.shape)

    som = KohonenSOM(grid_size=(10, 10), input_dim=15)

    print("Training SOM (rectangular)...")
    som.train_rectangular(train_data)

    print("\nTest sample mappings:")
    neuron_labels = label_som(som, train_data, train_labels)
    correct = 0

    print("\nTest predictions:\n")
    for x, true_label in zip(test_data, test_labels):
        pred = predict(som, neuron_labels, x)
        print(f"true={true_label:15} predicted={pred}")

        if pred == true_label:
            correct += 1

    accuracy = correct / len(test_labels)
    print("\nAccuracy:", accuracy)

    
    # Here should be the same for WTA


    print("\n\nLearning rate effectiveness analysis (rectangular):")
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
    lr_results = analyze_effectiveness_from_lr_rectangular(train_data, train_labels, test_data, test_labels, learning_rates)

    plt.plot(learning_rates, lr_results, marker="o")
    plt.xlabel("Learning rate")
    plt.ylabel("Classification error")
    plt.title("Error vs Learning Rate (Rectangular)")
    plt.show()


    # Here should be the same for WTA


    # Here should be the comparison of algorithms


    print("\n\nTraining dataset size analysis (rectangular):")
    train_ns = range(5, 50, 5)
    size_results = analyze_effectiveness_from_size_rectangular(train_ns, 2)

    plt.plot(train_ns, size_results, marker="o")
    plt.xlabel("Training dataset size by class")
    plt.ylabel("Classification error")
    plt.title("Error vs Training Set Size (Rectangular)")
    plt.show()


    # Here should be the same for WTA