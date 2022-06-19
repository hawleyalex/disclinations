import numpy as np


class Island:
    """
    Store information about an island, translate and rotate the island, and get the coordinates for the vertices.
    Each island is represented by a rectangle.
    """
    def __init__(self, center, length, width, theta, sigma=None):
        """
        Initialize island with position information.

        :param center: np array of int, center of the island in pixels
        :param length: float, length of the island in pixels
        :param width: float, length of the island in pixels
        :param theta: float, angle between x-axis and island length in radians
        :param sigma int, 1 indicates pointing toward right when theta=0, -1 indicates left when theta=0
        """
        self.center = center
        self.length = length
        self.width = width
        self.theta = theta  # theta = 0 means island is horizontal
        self.sigma = sigma
        return

    def translate(self, x, y):
        """
        Translate the island.

        :param x: float, amount to translate the island by in the x direction in pixels
        :param y: float, amount to translate the island by in the y direction in pixels
        :return: none
        """
        self.center = self.center + [x, y]
        return

    def rotate(self, rad, origin):
        """
        Rotate the island.

        :param rad: float, amount to rotate the island by in radians
        :param origin: np array of float, point to rotate the island about in pixels
        :return: none
        """
        # Change center
        center = self.center
        cos_th = np.cos(rad)
        sin_th = np.sin(rad)
        delta_x = center[0] - origin[0]
        delta_y = center[1] - origin[1]
        self.center[0] = origin[0] + cos_th * delta_x - sin_th * delta_y
        self.center[1] = origin[1] + sin_th * delta_x + cos_th * delta_y

        # Change theta
        self.theta = self.theta + rad
        return

    def scale(self, factor, origin):
        """
        Move the island towards/away from the origin by a factor.

        :param factor: float, factor by which island will be moved relative to origin
        :param origin: np array [x, y], location relative to which island location will be scaled
        :return: none
        """
        delta_x = self.center[0] - origin[0]
        delta_y = self.center[1] - origin[1]
        self.center[0] = delta_x * factor + origin[0]
        self.center[1] = delta_y * factor + origin[1]  # TODO: this can be written more compactly
        return

    def coords(self):
        """
        Get the coordinates for the four vertices of the island in the form [x0, y0, x1, y1...]

        :return: np array of float
        """
        sin_th = np.sin(self.theta)
        cos_th = np.cos(self.theta)

        arr = np.array([
            self.center[0] + cos_th * self.length / 2 - sin_th * self.width / 2,
            self.center[1] + sin_th * self.length / 2 + cos_th * self.width / 2,
            self.center[0] + cos_th * self.length / 2 + sin_th * self.width / 2,
            self.center[1] + sin_th * self.length / 2 - cos_th * self.width / 2,
            self.center[0] - cos_th * self.length / 2 + sin_th * self.width / 2,
            self.center[1] - sin_th * self.length / 2 - cos_th * self.width / 2,
            self.center[0] - cos_th * self.length / 2 - sin_th * self.width / 2,
            self.center[1] - sin_th * self.length / 2 + cos_th * self.width / 2,
        ])

        return arr
