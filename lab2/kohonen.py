import numpy as np

class KohonenSOM:
    def __init__(self, grid_size=(10, 10), input_dim=3, learning_rate=0.5, radius=None):
        self.width, self.height = grid_size
        self.input_dim = input_dim
        self.lr0 = learning_rate
        self.radius0 = radius if radius else max(grid_size) / 2

        self.weights = np.random.rand(self.width, self.height, input_dim)

    def _decay(self, initial, step, max_steps):
        return initial * np.exp(-step / max_steps)

    def _find_bmu(self, x):
        diff = self.weights - x
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def _rectangular_neighborhood(self, bmu_idx, radius):
        neigh = np.zeros((self.width, self.height))

        for i in range(self.width):
            for j in range(self.height):
                if abs(i - bmu_idx[0]) <= radius and abs(j - bmu_idx[1]) <= radius:
                    neigh[i, j] = 1

        return neigh

    def train_rectangular(self, data, epochs=10000):
        n_samples = data.shape[0]

        for step in range(epochs):
            x = data[np.random.randint(0, n_samples)]

            lr = self._decay(self.lr0, step, epochs)
            radius = int(self._decay(self.radius0, step, epochs))

            bmu_idx = self._find_bmu(x)

            neigh = self._rectangular_neighborhood(bmu_idx, radius)

            for i in range(self.width):
                for j in range(self.height):
                    if neigh[i, j] == 1:
                        self.weights[i, j] += lr * (x - self.weights[i, j])

    def train_wta(self, data, epochs=1000):
        #here should be the wta algorithm
        pass

    def map_vector(self, x):
        return self._find_bmu(x)