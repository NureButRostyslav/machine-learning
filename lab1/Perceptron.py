import random
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    """
    A single-layer perceptron for binary classification.
    Implements Rosenblatt's neuron model.
    """

    def __init__(self, n_inputs, learning_rate):
        """
        Initialize the perceptron with small random weights and bias.
        """
        if not (0 < learning_rate < 1):
            raise ValueError("Learning rate must be between 0 and 1")

        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        """
        Calculate the activation output for a given input vector.
        """
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        s = weighted_sum - self.bias

        return 1 if s >= 0 else -1

    def train(self, train_data, max_epochs=100):
        """
        Training of a perceptron using the Rosenblatt algorithm
        """
        epochs_passed = 0
        necessary_results = len(train_data)
        random.shuffle(train_data)
        while True:
            epochs_passed += 1
            if epochs_passed > max_epochs:
                print(f"Epochs trained: {max_epochs}")
                return
            
            correct_results = 0
            for x, t in train_data:
                res = sum(w * x for w, x in zip(self.weights, x)) - self.bias
                if t * res <= 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * t * x[i]
                    self.bias -= self.learning_rate * t

                    correct_results = 0
                else:
                    correct_results += 1

                    if correct_results == necessary_results:
                        print(f"Epochs required for training: {epochs_passed}")
                        return


def prepare_data():
    """
    Generate linearly separable data for two classes.
    Splits the dataset into training and testing sets (4:1 ratio).
    """
    dataset = []

    for _ in range(50):
        x1 = random.uniform(0.0, 0.4)
        x2 = random.uniform(0.0, 0.4)
        x3 = random.uniform(0.7, 1.0)
        dataset.append(([x1, x2, x3], 1))

    for _ in range(50):
        x1 = random.uniform(0.6, 1.0)
        x2 = random.uniform(0.6, 1.0)
        x3 = random.uniform(0.1, 0.3)
        dataset.append(([x1, x2, x3], -1))

    random.shuffle(dataset)

    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    return train_data, test_data


def evaluate_accuracy(perceptron_model, test_set):
    """
    Evaluate the model's accuracy on a separate testing dataset.
    Returns the percentage of correct predictions.
    """
    if not test_set or not perceptron_model:
        return 0.0

    correct_answers = 0

    for x, t in test_set:
        prediction = perceptron_model.predict(x)

        if t == prediction:
            correct_answers += 1

    return (correct_answers / len(test_set)) * 100

def generate_graph(perceptron_model, test_set):
    """
    Graphical representation of classification results.
    Displays sample points and the separating hyperplane H.
    """
    if not test_set or not perceptron_model:
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [point[0][0] for point in test_set]
    ys = [point[0][1] for point in test_set]
    zs = [point[0][2] for point in test_set]
    labels = [point[1] for point in test_set]

    colors = ['blue' if label == 1 else 'red' for label in labels]

    ax.scatter(xs, ys, zs, c=colors)

    w1, w2, w3 = perceptron_model.weights
    b = -perceptron_model.bias

    x_range = np.linspace(min(xs), max(xs), 10)
    y_range = np.linspace(min(ys), max(ys), 10)
    X, Y = np.meshgrid(x_range, y_range)

    if w3 != 0:
        Z = (-b - w1*X - w2*Y) / w3
        ax.plot_surface(X, Y, Z, alpha=0.3)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title("Perceptron")

    plt.show()


if __name__ == '__main__':
    train, test = prepare_data()
    model = Perceptron(n_inputs=3, learning_rate=0.1)

    initial_accuracy = evaluate_accuracy(model, test)
    print(f"Initial accuracy (random weights): {initial_accuracy:.2f}%")
    generate_graph(model, test)

    model.train(train)

    final_accuracy = evaluate_accuracy(model, test)
    print(f"Final accuracy (random weights): {final_accuracy:.2f}%")
    generate_graph(model, test)