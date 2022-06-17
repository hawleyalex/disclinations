import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tk


class Island:
    """
    Store information about an island, translate and rotate the island, and get the coordinates for the vertices.
    Each island is represented by a rectangle.
    """
    def __init__(self, center, length, width, theta):
        """
        Initialize island with position information.

        :param center: np array of int, center of the island in pixels
        :param length: float, length of the island in pixels
        :param width: float, length of the island in pixels
        :param theta: float, angle between x-axis and island length in radians
        """
        self.center = center
        self.length = length
        self.width = width
        self.theta = theta  # theta = 0 means island is horizontal
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
        self.theta = self.theta + rad  # TODO: this is very wrong
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

class MFM_GUI:
    """
    GUI for aligning islands to scans. Currently only produces tester islands to move/rotate.
    """
    def __init__(self, fp):
        """
        Initialize GUI.

        :param fp: string, filepath to image used to align islands
        """
        # Create tk window
        self.window = tk.Tk()
        self.window.geometry('720x600')  # TODO: make this adjustable/make sense

        # Load an image
        im = Image.open(fp)
        im_x, im_y = im.size
        self.img = ImageTk.PhotoImage(im)

        # Create canvas widget and add image
        self.canvas = tk.Canvas(self.window, width=im_x, height=im_y)
        self.canvas.create_image(10, 10, anchor='nw', image=self.img)
        self.canvas.pack()

        # Draw map of ideal island locations
        self.draw_map()

        # Set binds
        self.canvas.bind('<Button-1>', self.rec_cursor)
        self.canvas.bind('<B1-Motion>', self.move_map)
        self.canvas.bind('<Double-Button-1>', self.rotate_map)  # TODO: Create a button/better functionality

        # Run GUI
        self.window.mainloop()

        return

    def draw_map(self):
        # Load map from file

        # Calculate rectangle vertices
        length = 20  # TODO: These variables don't belong here
        width = 5
        self.islands = [  # TODO: Think about whether this belongs here
            Island(np.array([40, 40]), length, width, 0*np.pi/4),
            Island(np.array([80, 80]), length, width, 0*np.pi/6)
        ]

        # Set center
        self.center = np.array([60, 60])

        # Draw map
        self.rec_objects = [self.canvas.create_polygon(*island.coords(), outline='red') for island in self.islands]
        return

    def rec_cursor(self, event):
        """
        Record the position of the cursor at time of the event. Stores position in self.store_x and self.store_y.

        :param event: part of tkinter, automatically implemented through the bind mechanic
        :return: none
        """
        self.store_x = event.x  # TODO: These should be initialized somewhere else
        self.store_y = event.y
        return

    def move_map(self, event):  # TODO: rename to translate_map
        """
        Translate island map on canvas.

        :param event: tkinter, automatically implemented through bind
        :return:  none
        """
        # Calculate change in cursor position
        delta_x = event.x - self.store_x
        delta_y = event.y - self.store_y

        # Set new cursor position
        self.store_x = event.x
        self.store_y = event.y

        # Set new center
        self.center = self.center + [delta_x, delta_y]

        for i, island in enumerate(self.islands):
            self.canvas.delete(self.rec_objects[i])
            island.translate(delta_x, delta_y)
            self.rec_objects[i] = self.canvas.create_polygon(*island.coords(), outline='red', fill='')  # TODO: SPOT
        return

    def rotate_map(self, event):
        """
        Rotate island map on canvas.

        :param event: tkinter, automatically implemented through bind TODO: remove if unused
        :return: none
        """
        rad = np.pi/8

        for i, island in enumerate(self.islands):
            self.canvas.delete(self.rec_objects[i])

            island.rotate(rad, self.center)
            self.rec_objects[i] = self.canvas.create_polygon(*island.coords(), outline='red', fill='')

        return


# Testing!
# image_filepath = r'C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_1.spm.png'
# MFM_GUI(image_filepath)
