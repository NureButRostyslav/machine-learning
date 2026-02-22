import random


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
        Formula: y = step(sum(w_i * x_i) - b)
        """
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        s = weighted_sum - self.bias

        return 1 if s >= 0 else -1

    def train(self, train_data, epochs=100):
        """
        Implement the Rosenblatt learning rule here.
        Steps:
        1. Iterate through epochs.
        2. For each point in train_data, get the prediction.
        3. If prediction != expected, update self.weights and self.bias.
        """
        pass


def prepare_data():
    """
    Generate linearly separable data for two classes.
    Splits the dataset into training and testing sets (4:1 ratio).
    """
    dataset = []

    for _ in range(50):
        x1 = random.uniform(0.0, 0.4)
        x2 = random.uniform(0.0, 0.4)
        dataset.append([x1, x2, 1])

    for _ in range(50):
        x1 = random.uniform(0.6, 1.0)
        x2 = random.uniform(0.6, 1.0)
        dataset.append([x1, x2, -1])

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
    if not test_set:
        return 0.0

    correct_answers = 0

    for point in test_set:
        coordinates = point[:2]
        true_label = point[2]
        prediction = perceptron_model.predict(coordinates)

        if true_label == prediction:
            correct_answers += 1

    return (correct_answers / len(test_set)) * 100


if __name__ == '__main__':
    train, test = prepare_data()
    model = Perceptron(n_inputs=2, learning_rate=0.1)

    initial_accuracy = evaluate_accuracy(model, test)
    print(f"Initial accuracy (random weights): {initial_accuracy:.2f}%")

    # call model.train(train) here