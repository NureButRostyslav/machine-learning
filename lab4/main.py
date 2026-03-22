from hopfield import HopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt
import time

def compute_thresholds(data):
    """
    data: (samples, features)
    """
    return np.mean(data, axis=0)

def binarize(data, thresholds):
    return np.where(data >= thresholds, 1, -1)

def generate_class_samples(mean, std, n):
    """Generate samples for a class"""
    return np.random.normal(mean, std, (n, len(mean)))
    # noisy_std = np.array(std) * 5.0
    # return np.random.normal(mean, noisy_std, (n, len(mean)))

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
        "passenger_ferry": {
            "mean": [180, 28, 7, 35000, 28, 22, 5000, 2000, 120, 3000, 25000, 90, 10, 6, 8],
            "std":  [10, 2, 1, 4000, 2, 2, 1000, 200, 10, 300, 2000, 6, 2, 1, 1],
        },
        # "fishing_vessel": {
        #     "mean": [45, 10, 4, 900, 16, 12, 100, 10, 12, 200, 1500, 30, 15, 5, 7],
        #     "std":  [5, 1, 0.5, 150, 2, 2, 40, 4, 2, 40, 200, 5, 2, 1, 1],
        # }
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

    all_data = np.vstack((train_data, test_data))
    thresholds = compute_thresholds(all_data)

    all_data = binarize(all_data, thresholds)

    train_data = all_data[:len(train_data)]
    test_data = all_data[len(train_data):]

    train_data, train_labels = shuffle_data(train_data, train_labels)
    test_data, test_labels = shuffle_data(test_data, test_labels)

    return train_data, train_labels, test_data, test_labels

def compare_hopfield_algorithms(patterns, test_data, test_labels, pattern_labels):
    print("\n--- Algorithm Comparison ---")

    # 1. Hebb
    hopfield_hebb = HopfieldNetwork()
    start_time = time.time()
    hopfield_hebb.train_hebb(patterns)
    hebb_time = time.time() - start_time

    correct_hebb = 0
    for x, t in zip(test_data, test_labels):
        pred = closest_pattern(hopfield_hebb.predict(x), patterns, pattern_labels)
        if pred == t: correct_hebb += 1
    err_hebb = 1.0 - (correct_hebb / len(test_labels))

    # 2. Projection
    hopfield_proj = HopfieldNetwork()
    start_time = time.time()
    hopfield_proj.train_projection(patterns)
    proj_time = time.time() - start_time

    correct_proj = 0
    for x, t in zip(test_data, test_labels):
        pred = closest_pattern(hopfield_proj.predict(x), patterns, pattern_labels)
        if pred == t: correct_proj += 1
    err_proj = 1.0 - (correct_proj / len(test_labels))

    print(f"Hebb:       Error = {err_hebb:.4f}, Time = {hebb_time:.6f} sec")
    print(f"Projection: Error = {err_proj:.4f}, Time = {proj_time:.6f} sec")

    algorithms = ['Hebb', 'Projection']
    errors = [err_hebb, err_proj]
    times = [hebb_time, proj_time]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(x - width / 2, errors, width, label='Classification Error', color='skyblue')
    ax1.set_ylabel('Classification Error', color='blue')
    max_err = max(errors) if max(errors) > 0 else 0.1
    ax1.set_ylim(0, max_err + 0.1)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, times, width, label='Execution Time (s)', color='salmon')
    ax2.set_ylabel('Time (seconds)', color='red')
    ax2.set_ylim(0, max(max(times) * 1.2, 0.001))
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title('Algorithm Comparison: Error vs Execution Time')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

def analyze_accurace_by_noise_proj(hopfield, test_data, test_labels, patterns, pattern_labels):
    noise_levels = np.linspace(0, 1, 5)
    results = []

    for noise in noise_levels:
        noisy_test = add_noise(test_data, noise)
        correct = 0

        for x, true_label in zip(noisy_test, test_labels):
            restored = hopfield.predict(x)
            pred_label = closest_pattern(restored, patterns, pattern_labels)
            if pred_label == true_label:
                correct += 1

        accuracy = correct / len(test_labels)
        results.append((noise, accuracy))

        print(f"Noise level: {noise:.2f} | Accuracy: {accuracy:.4f}")

    return results

def analyze_learning_rate(patterns, test_data, test_labels, pattern_labels):
    print("\n--- Analyzing Learning Rate (Hebb) ---")
    rates = [0.1, 0.5, 1.0, 2.0, 5.0]
    accuracies = []

    for lr in rates:
        hopfield = HopfieldNetwork()
        hopfield.train_hebb(patterns, learning_rate=lr)

        correct = 0
        for x, true_label in zip(test_data, test_labels):
            restored = hopfield.predict(x)
            if closest_pattern(restored, patterns, pattern_labels) == true_label:
                correct += 1
        accuracies.append(correct / len(test_labels))

    # Малювання графіка має бути ТУТ, всередині функції:
    plt.figure()
    plt.plot(rates, accuracies, marker="o", color="green")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Learning Rate (Hebb)")
    plt.ylim(0, 1.1)
    plt.show()

def closest_pattern(output, patterns, labels):
    distances = [np.sum(output != p) for p in patterns]
    return labels[np.argmin(distances)]

def analyze_accurace_by_noise_hebb(hopfield, test_data, test_labels, patterns, pattern_labels):
    noise_levels = np.linspace(0, 1, 5)
    results = []

    for noise in noise_levels:
        noisy_test = add_noise(test_data, noise)

        correct = 0

        for x, true_label in zip(noisy_test, test_labels):
            restored = hopfield.predict(x)
            pred_label = closest_pattern(restored, patterns, pattern_labels)

            if pred_label == true_label:
                correct += 1

        accuracy = correct / len(test_labels)
        results.append((noise, accuracy))

        print(f"Noise level: {noise:.2f}")
        print(f"Accuracy: {accuracy:.4f}")

    return results

def add_noise(data, noise_level):
    """
    noise_level: від 0 до 1 (частка спотворених бітів)
    """
    noisy_data = data.copy()
    n_samples, n_features = data.shape

    for i in range(n_samples):
        flip_mask = np.random.rand(n_features) < noise_level
        noisy_data[i][flip_mask] *= -1  # спотворення

    return noisy_data

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = generate_dataset(20, 3)

    unique_labels = list(set(train_labels))
    patterns = []
    pattern_labels = []

    for label in unique_labels:
        idx = list(train_labels).index(label)
        patterns.append(train_data[idx])
        pattern_labels.append(label)

    patterns = np.array(patterns)

    print("Training the network (Hebb)...")

    hopfield = HopfieldNetwork()
    hopfield.train_hebb(patterns)

    correct = 0

    print("Test predictions:\n")
    for i, (x, true_label) in enumerate(zip(test_data, test_labels)):
        restored = hopfield.predict(x)
        pred_label = closest_pattern(restored, patterns, pattern_labels)

        is_correct = pred_label == true_label

        print(f"Sample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted : {pred_label}")
        print(f"  Correct   : {is_correct}")
        print("-" * 30)

        if is_correct:
            correct += 1

    accuracy = correct / len(test_labels)
    print(f"\nFinal Accuracy: {accuracy:.4f}")

    
    # Here should be the same for projection
    print("\nTraining the network (Projection)...")
    hopfield_proj = HopfieldNetwork()
    hopfield_proj.train_projection(patterns)

    correct_proj = 0
    print("Test predictions (Projection):\n")
    for i, (x, true_label) in enumerate(zip(test_data, test_labels)):
        restored = hopfield_proj.predict(x)
        pred_label = closest_pattern(restored, patterns, pattern_labels)
        is_correct = pred_label == true_label

        print(f"Sample {i + 1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted : {pred_label}")
        print(f"  Correct   : {is_correct}")
        print("-" * 30)

        if is_correct:
            correct_proj += 1

    accuracy_proj = correct_proj / len(test_labels)
    print(f"\nFinal Accuracy (Projection): {accuracy_proj:.4f}")
    
    # There is no learning rate


    # Here should be the comparison of algorithms
    compare_hopfield_algorithms(patterns, test_data, test_labels, pattern_labels)

    print("\n\nAccuracy worsening due to noise analysis (Hebb):")
    noise_results_hebb = analyze_accurace_by_noise_hebb(hopfield, test_data, test_labels, patterns, pattern_labels)

    print("\n\nAccuracy worsening due to noise analysis (Projection):")
    noise_results_proj = analyze_accurace_by_noise_proj(hopfield_proj, test_data, test_labels, patterns, pattern_labels)

    plt.figure()
    plt.plot([r[0] for r in noise_results_hebb], [r[1] for r in noise_results_hebb], marker="o", label="Hebb Rule")
    plt.plot([r[0] for r in noise_results_proj], [r[1] for r in noise_results_proj], marker="s",
             label="Projection Rule", color="orange")
    plt.xlabel("Noise level")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Noise level")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # print("\n\nAccuracy worsening due to noice analysis (Hebb):")
    # noice_recults_hebb = analyze_accurace_by_noise_hebb(hopfield, test_data, test_labels, patterns, pattern_labels)
    #
    # plt.figure()
    # plt.plot([r[0] for r in noice_recults_hebb], [r[1] for r in noice_recults_hebb], marker="o")
    # plt.xlabel("Noise level")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy VS Noise level (Hebb)")
    # plt.show()
    #
    # print("\n\nAccuracy worsening due to noise analysis (Projection):")
    # noise_results_proj = analyze_accurace_by_noise_proj(hopfield_proj, test_data, test_labels, patterns, pattern_labels)
    #
    # plt.figure()
    # plt.plot([r[0] for r in noise_results_proj], [r[1] for r in noise_results_proj], marker="s", color="orange")
    # plt.xlabel("Noise level")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy VS Noise level (Projection)")
    # plt.show()

    analyze_learning_rate(patterns, test_data, test_labels, pattern_labels)