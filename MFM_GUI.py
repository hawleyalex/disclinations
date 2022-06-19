import numpy as np
from PIL import Image
from PIL import ImageTk
from PIL import ImageEnhance
import tkinter as tk
from Island import Island
import MFM_Arrows
from matplotlib.path import Path


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
        self.fp = fp
        self.ilocs = ilocs

        # Load an image
        self.im = Image.open(fp).convert('L')
        enhancer = ImageEnhance.Contrast(self.im)
        self.im_contrast = enhancer.enhance(3)
        im_x, im_y = self.im.size
        self.img = ImageTk.PhotoImage(self.im)

        # Create canvas widget and add image
        self.canvas = tk.Canvas(self.window, width=im_x, height=im_y)
        self.canvas.create_image(0, 0, anchor='nw', image=self.img)
        self.canvas.grid(row=1, column=3, rowspan=5)

        # Create buttons
        self.rotate_button = tk.Button(self.window, text="Rotate", command=self.rotate_map)
        self.rotate_button.grid(row=1, column=1)
        self.scale_button = tk.Button(self.window, text="Scale", command=self.click_scale_button)
        self.scale_button.grid(row=2, column=1)

        self.scale_text = tk.StringVar()
        self.scale_entry = tk.Entry(self.window, textvariable=self.scale_text)
        self.scale_entry.grid(row=3, column=1)

        self.excise_button = tk.Button(self.window, text="Excise", command=self.draw_excised)
        self.excise_button.grid(row=4, column=1)

        self.sigmas_button = tk.Button(self.window, text="Arrows", command=self.draw_sigmas)
        self.sigmas_button.grid(row=5, column=1)

        # Draw map of ideal island locations
        self.draw_map()

        # Set binds
        self.canvas.bind('<Button-1>', self.rec_cursor)
        self.canvas.bind('<B1-Motion>', self.translate_map_curs)
        self.canvas.bind('<Double-Button-1>', self.rotate_map)  # TODO: Create a button/better functionality
        self.window.bind('<Key>', self.key_press)

        # Set state variable
        self.select_mode = False
        self.current_select = None
        self.current_ind = None

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

        self.islands = [  # TODO: initialize in __init__
            Island(np.array([row[0], row[1]]), length, width, row[2]) for row in island_data
        ]

        # Set center
        self.center = np.array([np.mean(island_data[:,0]), np.mean(island_data[:,1])])

        # Draw map
        self.rec_objects = [self.canvas.create_polygon(*island.coords(), outline='red', fill='') for island in self.islands]
        return

    def draw_excised(self):
        """
        Calculate the coordinates of the rectangles that will be excised to represent the poles of each island.
        Plot the excised rectangles on the canvas.

        :return: none
        """
        scale_factor = 1/2  # TODO: magic number, make this a parameter

        inner_boxes = [
            Island(island.center, island.length * scale_factor, island.width, island.theta) for island in self.islands
        ]
        # TODO: all of these class variables should be defined in __init___
        self.excised_coords1 = [
            np.array([*inner.coords()[:4], *outer.coords()[2:4], *outer.coords()[:2]])
            for inner, outer in zip(inner_boxes, self.islands)
        ]

        self.excised_coords2 = [
            np.array([*inner.coords()[4:], *outer.coords()[6:], *outer.coords()[4:6]])
            for inner, outer in
            zip(inner_boxes, self.islands)
        ]

        self.rec_excised1 = [
            self.canvas.create_polygon(
                *coords,
                outline='green', fill='')
            for coords in self.excised_coords1
        ]
        self.rec_excised2 = [
            self.canvas.create_polygon(
                *coords,
                outline='green', fill='')
            for coords in self.excised_coords2
        ]
        return

    def draw_sigmas(self):
        """
        Draw the direction that the nanomagnets are pointing in on the canvas.

        :return: none
        """
        self.calculate_sigmas()

        self.arrows = [
            self.canvas.create_line(
                *island.center,
                island.center[0]+np.cos(island.theta + np.pi * (island.sigma-1)/-2)*island.length/5,
                island.center[1]+np.sin(island.theta + np.pi * (island.sigma-1)/-2)*island.length/5,
                arrow=tk.LAST, fill='red'
            )
            for island in self.islands
        ]
        return

    def calculate_sigmas(self):
        """
        Using the current island positions, calculate the sigma of each island.

        :return: none
        """
        # IMPORTANT: PIL uses column-major, so this array has to be indexed as such.
        im_arr = np.asarray(self.im_contrast)
        for i, island in enumerate(self.islands):
            x1, y1 = self.get_inds(self.excised_coords1[i])
            x2, y2 = self.get_inds(self.excised_coords2[i])

            av1 = np.mean(im_arr[y1, x1])  # indexing using column-major
            av2 = np.mean(im_arr[y2, x2])

            if av1 > av2:
                island.sigma = 1
                # self.canvas.create_polygon(*self.excised_coords1[i], outline='green', fill='')
            elif av2 > av1:
                island.sigma = -1
                # self.canvas.create_polygon(*self.excised_coords2[i], outline='green', fill='')
            else:
                island.sigma = 0
        return

    def get_inds(self, rec):
        """
        Given the coordinates of a rectangle, return the indices of all points that lie within the rectangle.

        NOTE: I was not picky about whether edges are "within" the rectangle. Since these points will be
        averaged, a single row of pixels shouldn't matter. If I'm wrong, then start by looking at matplotlib's
        contains_points function which has some documented inconsistent behavior with edges.

        :param rec: list-like, The coordinates of the four corners of an island, [x0, y0,...,x3, y3]
        :return: xs, ys, both 1d numpy arrays containing the x and y coordinates of all points inside
        of the island.
        """
        # Get coordinates for a rectangle that circumscribes the given rectangle
        min_x, max_x = int(min(rec[::2])), int(max(rec[::2]))
        min_y, max_y = int(min(rec[1::2])), int(max(rec[1::2]))

        # This draws the upright rectangles that will be "cut from," can be useful for testing
        # You can also easily crop the PIL image or index sections of the numpy array
        # I kept these calculation functions as methods because it's FAR easier to debug.
        # more easily with these rectangles.
        # self.canvas.create_polygon(
        #     min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y,
        #     outline='pink', fill=''
        # )

        # You'll see a lot of strange things happening with switching x and y positions:
        # this is because of the way different numpy functions behave; check the docs if
        # you think something is wrong.

        # Create a grid for a rectangle that circumscribes the given rectangle
        x, y = np.meshgrid(np.arange(max_x - min_x + 1), np.arange(max_y - min_y + 1))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        # Shift the given rectangle's coordinates so that the origin is at (min_x, min_y)
        shift_rec = [
            (rec[0] - min_x, rec[1] - min_y),
            (rec[2] - min_x, rec[3] - min_y),
            (rec[4] - min_x, rec[5] - min_y),
            (rec[6] - min_x, rec[7] - min_y),
        ]

        p = Path(shift_rec)  # draw the island within the rectangle
        grid = p.contains_points(points)
        mask = grid.reshape(max_y - min_y + 1, max_x - min_x + 1)  # make bool mask for points inside the given rect
        inds = np.nonzero(mask)  # get the indices of the points within the given rectangle
        xs = inds[1] + min_x  # return to the original coordinate system
        ys = inds[0] + min_y

        return xs, ys

    def rec_cursor(self, event):
        """
        Record the position of the cursor at time of the event. Stores position in self.store_x and self.store_y.

        :param event: part of tkinter, automatically implemented through the bind mechanic
        :return: none
        """
        self.store_x = event.x  # TODO: These should be initialized somewhere else
        self.store_y = event.y
        return

    def translate_map_curs(self, event):
        """
        Translate island map on canvas using cursor.

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

    def translate_map(self, x, y):
        """
        Translates map in number of pixels specified.

        :param x: int, pixels to translate in the x direction
        :param y: int, pixels to tranlate in the y direction
        :return: none
        """
        # Set new center
        self.center = self.center + [x, y]

        for i, island in enumerate(self.islands):
            self.canvas.delete(self.rec_objects[i])
            island.translate(x, y)
            self.rec_objects[i] = self.canvas.create_polygon(*island.coords(), outline='red', fill='')  # TODO: SPOT
        return

    def translate_island(self, x, y, event=None):
        """
        Translates selected island to the right while in select mode.

        :param x: int, pixels to translate in the x direction
        :param y: int, pixels to tranlate in the y direction
        :param event: tkinter variable, not necessary
        :return: none
        """

        i = self.current_ind
        self.canvas.delete(self.rec_objects[i])
        self.current_select.translate(x, y)
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
        if self.current_ind is None:
            self.current_ind = 0
        self.current_select = self.islands[self.current_ind]

        # Color selected island yellow
        self.canvas.delete(self.rec_objects[self.current_ind])
        self.rec_objects[self.current_ind] = self.canvas.create_polygon(*self.current_select.coords(), outline='yellow', fill='')
        return

    def deactivate_select(self):
        """
        Removes color of current selection.

        :return: none
        """
        # Color selected island yellow
        self.canvas.delete(self.rec_objects[self.current_ind])
        self.rec_objects[self.current_ind] = self.canvas.create_polygon(*self.current_select.coords(), outline='red',
                                                                        fill='')
        return

    def change_select(self, forward=True):
        """
        Changes current selection to next on list and colors that selection yellow.

        :param forward: bool, if True, then selection moves forward, else moves backward
        :return: none
        """
        # Color deselected island red
        self.canvas.delete(self.rec_objects[self.current_ind])
        self.rec_objects[self.current_ind] = self.canvas.create_polygon(*self.current_select.coords(), outline='red', fill='')

        # Move selection
        if forward:
            self.current_ind += 1
        else:
            self.current_ind -= 1

        i = self.current_ind
        try:
            self.current_select = self.islands[i]
        except IndexError:
            if forward:
                self.current_ind = 0
            else:
                self.current_ind = len(self.islands) - 1

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

    def test_calc(self):
        MFM_Arrows.find_directions(self.islands, 0, self.fp)
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
                self.select_mode = False
                self.deactivate_select()
            else:
                self.select_mode = True
                self.activate_select()
        elif char == "c" and self.select_mode:
            self.change_select(forward=True)
        elif char == "z" and self.select_mode:
            self.change_select(forward=False)

        # Move island(s)
        elif char == "w":  # TODO: might be better to map this to arrow keys
            if self.select_mode:
                self.translate_island(0, -1)
            else:
                self.translate_map(0, -1)
        elif char == "a":
            if self.select_mode:
                self.translate_island(-1, 0)
            else:
                self.translate_map(-1, 0)
        elif char == "s":
            if self.select_mode:
                self.translate_island(0, 1)
            else:
                self.translate_map(0, 1)
        elif char == "d":
            if self.select_mode:
                self.translate_island(1, 0)
            else:
                self.translate_map(1, 0)

        # Save island locations
        elif char == "q":
            self.save_ilocs()

        elif char == "h":
            self.test_calc()

        return

# Testing!
# image_filepath = r'C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_1.spm.png'
# image_filepath = r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_5.spm.png"
# ilocs = r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\ilocs0_00334_1.txt"
# MFM_GUI(image_filepath, ilocs)

