import numpy as np
import matplotlib.pyplot as plt

N_ANCHORS = 4
N_POINTS = 256


def generate_gradients(n_anchors):
    """Generate gradients of unit length"""
    thetas = 2 * np.pi * np.random.rand(n_anchors, n_anchors)
    gradients = np.stack((np.cos(thetas), np.sin(thetas)), axis=-1)
    return gradients


def lerp(a, b, u):
    """Smoothed interpolation"""
    uu = 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3
    return a + uu * (b - a)


def perlin(i, j, gradients):
    # Translate grid coordinates to anchor coordinates
    u, ii = np.modf(i / N_POINTS * (N_ANCHORS - 1))
    v, jj = np.modf(j / N_POINTS * (N_ANCHORS - 1))
    ii, jj = int(ii), int(jj)

    # Compute the dot products
    g1 = gradients[ii, jj] @ (u, v)
    g2 = gradients[ii + 1, jj] @ (u - 1, v)
    g3 = gradients[ii, jj + 1] @ (u, v - 1)
    g4 = gradients[ii + 1, jj + 1] @ (u - 1, v - 1)

    # Perform the interpolation
    v1 = lerp(g1, g2, u)
    v2 = lerp(g3, g4, u)
    value = lerp(v1, v2, v)

    return value


if __name__ == '__main__':
    gradients = generate_gradients(N_ANCHORS)

    grid = np.zeros((N_POINTS, N_POINTS))
    for i in range(N_POINTS):
        for j in range(N_POINTS):
            grid[i, j] = perlin(i, j, gradients)

    plt.imshow(grid, cmap="gray")
    plt.show()
