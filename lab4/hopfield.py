import numpy as np

class HopfieldNetwork:
    def __init__(self):
        self.W = None

    def train_hebb(self, patterns, learning_rate=1.0):
        P, N = patterns.shape
        self.W = np.zeros((N, N))

        for p in patterns:
            self.W += learning_rate * np.outer(p, p)
        np.fill_diagonal(self.W, 0)

    def train_projection(self, patterns):
        overlap_matrix = np.dot(patterns, patterns.T)
        try:
            inv_overlap = np.linalg.inv(overlap_matrix)
        except np.linalg.LinAlgError:
            inv_overlap = np.linalg.pinv(overlap_matrix)
        self.W = np.dot(np.dot(patterns.T, inv_overlap), patterns)
        np.fill_diagonal(self.W, 0)

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def update(self, state):
        # s_j = sum(w_ij * y_i)
        s = np.dot(self.W, state)
        # y_j = f(s_j)
        return self.activation(s)

    def predict(self, pattern, max_steps=100):
        state = pattern.copy()
        for _ in range(max_steps):
            new_state = self.update(state)
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state