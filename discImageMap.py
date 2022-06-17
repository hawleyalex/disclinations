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

class MFM_GUI:
    """
    GUI for aligning islands to scans. Currently only produces tester islands to move/rotate.
    """
    def __init__(self, fp, ilocs=None):
        """
        Initialize GUI.

        :param fp: string, filepath to image used to align islands
        :param ilocs: string, filepath to text file with island locations
        """
        # Create tk window
        self.window = tk.Tk()
        self.window.geometry('720x600')  # TODO: make this adjustable/make sense

        # Initialize variables
        self.ilocs = ilocs

        # Load an image
        im = Image.open(fp)
        im_x, im_y = im.size
        self.img = ImageTk.PhotoImage(im)

        # Create canvas widget and add image
        self.canvas = tk.Canvas(self.window, width=im_x, height=im_y)
        self.canvas.create_image(10, 10, anchor='nw', image=self.img)
        self.canvas.pack()

        # Create buttons
        self.rotate_button = tk.Button(self.window, text="Rotate", command=self.rotate_map)
        self.rotate_button.pack()
        self.scale_button = tk.Button(self.window, text="Scale", command=self.click_scale_button)
        self.scale_button.pack()

        self.scale_text = tk.StringVar()
        self.scale_entry = tk.Entry(self.window, textvariable=self.scale_text)
        self.scale_entry.pack()

        # Draw map of ideal island locations
        self.draw_map()

        # Set binds
        self.canvas.bind('<Button-1>', self.rec_cursor)
        self.canvas.bind('<B1-Motion>', self.move_map)
        self.canvas.bind('<Double-Button-1>', self.rotate_map)  # TODO: Create a button/better functionality
        self.window.bind('<Key>', self.key_press)

        # Set state variable
        self.select_mode = False
        self.current_select = None
        self.current_ind = 0

        # Run GUI
        self.window.mainloop()

        return

    def draw_map(self):
        """
        Draw outlines of islands onto canvas.

        :return: none
        """
        # Load map from file

        # Calculate rectangle vertices
        length = 20  # TODO: These variables don't belong here
        width = 10

        if self.ilocs:
            island_data = np.genfromtxt(self.ilocs,
                                        delimiter='\t')
        else:
            island_data = np.genfromtxt(r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\size2even.txt",
                                        delimiter='\t')

        self.islands = [  # TODO: Think about whether this belongs here
            Island(np.array([row[0], row[1]]), length, width, row[2]) for row in island_data
        ]

        # Set center
        self.center = np.array([np.mean(island_data[:,0]), np.mean(island_data[:,1])])

        # Draw map
        self.rec_objects = [self.canvas.create_polygon(*island.coords(), outline='red', fill='') for island in self.islands]
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

    def translate_island_right(self, event=None):
        """
        Moves selected island to the right while in select mode.

        :param event: tkinter variable, not necessary
        :return: none
        """
        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])
        self.current_select.translate(1,0)
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def translate_island_left(self, event=None):
        """
        Moves selected island to the left while in select mode.

        :param event: tkinter variable, not necessary
        :return: none
        """
        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])
        self.current_select.translate(-1,0)
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def translate_island_up(self, event=None):
        """
        Moves selected island up while in select mode.

        :param event: tkinter variable, not necessary
        :return: none
        """
        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])
        self.current_select.translate(0,-1)
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def translate_island_down(self, event=None):
        """
        Moves selected island down while in select mode.

        :param event: tkinter variable, not necessary
        :return: none
        """
        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])
        self.current_select.translate(0,1)
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def rotate_map(self, rad=np.pi/64, event=None):
        """
        Rotate island map on canvas about the map's center.

        :param rad: float, angle at which to rotate map in radians
        :return: none
        """

        for i, island in enumerate(self.islands):
            self.canvas.delete(self.rec_objects[i])

            island.rotate(rad, self.center)
            self.rec_objects[i] = self.canvas.create_polygon(*island.coords(), outline='red', fill='')
        return

    def rotate_island(self, rad=np.pi/64, event=None):
        """
        Rotate selected island about the ISLAND'S center by rad while in select mode.

        :param rad: float, angle at which to rotate island in radians
        :param event: tkinter variable, not necessary
        :return: none
        """

        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])

        self.current_select.theta += rad
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def scale_map(self, factor):
        """
        Scales island map by factor about center.

        :param factor: float, factor by which to scale island map
        :return: none
        """
        for i, island in enumerate(self.islands):
            self.canvas.delete(self.rec_objects[i])

            island.scale(factor, self.center)
            self.rec_objects[i] = self.canvas.create_polygon(*island.coords(), outline='red', fill='')

        return

    def activate_select(self):
        """
        Initializes current selection and colors it yellow.

        :return: none
        """
        # Set index and current island
        self.current_ind = 0
        self.current_select = self.islands[0]

        # Color selected island yellow
        self.canvas.delete(self.rec_objects[0])
        self.rec_objects[0] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def change_select(self):  # TODO: add behavior to go back and to loop around
        """
        Changes current selection to next on list and colors that selection yellow.

        :return: none
        """
        # Color deselected island red
        self.canvas.delete(self.rec_objects[self.current_ind])
        self.rec_objects[self.current_ind] = self.canvas.create_polygon(*self.current_select.coords(), outline='red', fill='')

        # Move selection
        self.current_ind += 1
        i = self.current_ind
        self.current_select = self.islands[i]

        # Color selected island yellow
        self.canvas.delete(self.rec_objects[i])
        self.rec_objects[i] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def click_scale_button(self):
        """
        Get text from scale text box. TODO: This will probably be discarded.

        :return: none
        """
        factor = self.scale_text.get()
        self.scale_map(float(factor))
        return

    def save_ilocs(self):
        write_string = ''
        for i, island in enumerate(self.islands):
            if i == 0:
                write_string = write_string + str(island.center[0]) + "\t" + str(island.center[1]) + "\t" + str(island.theta)
            else:
                write_string = write_string + "\n" + str(island.center[0]) + "\t" + str(island.center[1]) + "\t" + str(island.theta)
        with open(self.ilocs, "w+") as f:
            f.write(write_string)
        return

    def key_press(self, event):
        """
        Makes GUI responsive to keyboard.

        :param event: tkinter variable, not necessary
        :return: none
        """
        char = event.char
        # Rotate island(s)
        if char == "r":
            if self.select_mode:
                self.rotate_island(np.pi/64)
            else:
                self.rotate_map(np.pi/64)
        elif char == "e":
            if self.select_mode:
                self.rotate_island(-np.pi / 64)
            else:
                self.rotate_map(-np.pi / 64)

        # Scale island map (does nothing in select mode; islands should all be the same size)
        elif char == "=":
            self.scale_map(1.01)
        elif char == "-":
            self.scale_map(0.99)

        # Activate/deactivate select mode, change selection
        elif char == "x":
            if self.select_mode:
                self.select_mode = False  # TODO: add function for deactivating select mode
            else:
                self.select_mode = True
                self.activate_select()
        elif char == "c" and self.select_mode:
            self.change_select()

        # Move island(s)  TODO: add functionality for this in NOT select mode
        elif char == "w":  # TODO: might be better to map this to arrow keys
            if self.select_mode:
                self.translate_island_up()
        elif char == "a":
            if self.select_mode:
                self.translate_island_left()
        elif char == "s":
            if self.select_mode:
                self.translate_island_down()
        elif char == "d":
            if self.select_mode:
                self.translate_island_right()

        # Save island locations
        elif char == "q":
            self.save_ilocs()
        return

# Testing!
# image_filepath = r'C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_1.spm.png'
# image_filepath = r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_5.spm.png"
# ilocs = r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\ilocs0_00334_1.txt"
# MFM_GUI(image_filepath, ilocs)
