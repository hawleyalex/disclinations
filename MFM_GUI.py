import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from matplotlib import cm
from matplotlib.path import Path
from PIL import Image, ImageTk, ImageEnhance

from Island import Island
from GUI_Helpers import PointList
import MFM_Morph



class MFM_GUI:
    """
    GUI for aligning islands to scans.
    """
    def __init__(self, fp1, fp2, n, ndisc='single', ilocs=None, save_file=None):
        """
        Initialize GUI.

        :param fp1: string, filepath to image used to align islands
        :param fp2: string, filepath to image used to determine island orientations
        :param n: int, number of islands on "side" of square
        :param ndisc: string, number of disclinations. 'single' or 'double'
        :param ilocs: string, filepath to text file with island locations
        """
        # Create tk window
        self.window = tk.Tk()
        self.window.geometry('720x600')  # TODO: make this adjustable/make sense

        # Initialize variables
        self.fp1 = fp1
        self.fp2 = fp2
        self.n_side = n
        self.ndisc = ndisc
        self.ilocs = ilocs
        self.save_file = save_file

        n_islands_dict = {
            '05': 18,
            '06': 30,
            '09': 84,
            '10': 108,
            '13': 198,
            '14': 234,
            '19': 459,
            '20': 513
        }

        self.n_islands = n_islands_dict['{:0>2d}'.format(self.n_side)]

        # Find position file
        self.ideal_fp = r"C:/Users/sh2547/Documents/lab/islandcoordinates/{}{:0>2d}.txt".format(self.ndisc, self.n_side)

        # Load images, convert to black and white
        self.im1 = Image.open(fp1).convert('L')
        self.im2 = Image.open(fp2).convert('L')

        enhancer = ImageEnhance.Contrast(self.im1)
        self.im_contrast = enhancer.enhance(3)
        im_x, im_y = self.im1.size

        self.img1 = ImageTk.PhotoImage(self.im1)
        self.img2 = ImageTk.PhotoImage(self.im2)

        # Create canvas widget and add images
        self.canvas = tk.Canvas(self.window, width=im_x, height=im_y)
        self.image2_canvas_object = self.canvas.create_image(0, 0, anchor='nw', image=self.img2)
        self.image1_canvas_object = self.canvas.create_image(0, 0, anchor='nw', image=self.img1)
        self.canvas.grid(row=2, column=2, rowspan=7)

        self.switch_image_frame = tk.Frame(self.window)
        self.switch_image_frame.grid(row=1, column=2)

        self.image_1_button = tk.Button(self.switch_image_frame, text="Im1", command=self.image_1_button_click)
        self.image_1_button.grid(row=0, column=0, sticky='ne')
        self.image_2_button = tk.Button(self.switch_image_frame, text="Im2", command=self.image_2_button_click)
        self.image_2_button.grid(row=0, column=1, sticky='nw')

        self.build_guides_widgets()

        # Set state variable
        self.SET_GUIDES_MODE = 1
        self.SET_CENTERS_MODE = 2
        self.SET_ISLANDS_MODE = 3
        self.SET_SIMGAS_MODE = 4
        self.mode = self.SET_GUIDES_MODE
        self.select_mode = False
        self.current_select = None
        self.current_ind = None
        self.selected_center_ind = None
        self.selected_island_ind = None
        self.center_shapes = None

        # Initialize lists
        self.guide_points = PointList(self.canvas)

        # Draw map of ideal island locations
        self.draw_centers()

        # Set binds
        self.canvas.bind('<Button-1>', self.left_click)
        self.canvas.bind('<B1-Motion>', self.translate_map_curs)
        self.canvas.bind('<Button-3>', self.right_click)
        self.window.bind('<Key>', self.key_press)

        # Run GUI
        self.window.mainloop()

        return

    def build_guides_widgets(self):
        self.guides_instructions_label = tk.Label(
            self.window,
            text="""
            Place points at each of the 
            three center islands and 
            at all six 'corner' islands
            
            Right click to place
            Left click to select and move
            Press L to delete selection
            """, anchor=tk.NW, justify='left'
        )
        self.guides_instructions_label.grid(row=2, column=1, rowspan=3)

        self.guides_ok_button = tk.Button(self.window, text="Guides Ok", command=self.guides_ok)
        self.guides_ok_button.grid(row=5, column=1)

        self.guides_widgets = [
            self.guides_instructions_label,
            self.guides_ok_button
        ]

        return

    def build_centers_widgets(self):
        self.current_centers_var = tk.StringVar()
        self.thresh_entry_var = tk.StringVar()

        self.current_centers_label = tk.Label(self.window, textvariable=self.current_centers_var)
        self.current_centers_label.grid(row=2, column=1)

        self.centers_needed_label = tk.Label(self.window, text="Centers needed: {}".format(self.n_islands))
        self.centers_needed_label.grid(row=3, column=1)

        self.thresh_frame = tk.Frame(self.window)
        self.thresh_frame.grid(row=4, column=1, rowspan=2)

        self.set_thresh_label = tk.Label(self.thresh_frame, text="Set thresh: ")
        self.set_thresh_label.grid(row=1, column=1)

        self.set_thresh_entry = tk.Entry(self.thresh_frame, textvariable=self.thresh_entry_var)
        self.set_thresh_entry.grid(row=1, column=2)

        self.reset_thresh_button = tk.Button(self.thresh_frame, text="Reset thresh", command=self.draw_centers)
        self.reset_thresh_button.grid(row=2, column=1, columnspan=2)

        self.set_centers_instructions = tk.Label(
            self.window,
            text="""
            Click to select point
            Drag to move
            Right click to add point
            Press L to delete
            """
        )
        self.set_centers_instructions.grid(row=6, column=1)

        self.centers_ok_button = tk.Button(self.window, text="Centers Ok", command=self.centers_ok)
        self.centers_ok_button.grid(row=7, column=1)

        self.centers_widgets = [
            self.current_centers_label,
            self.centers_needed_label,
            self.thresh_frame,
            self.set_thresh_label,
            self.set_thresh_entry,
            self.reset_thresh_button,
            self.set_centers_instructions,
            self.centers_ok_button
        ]

        self.set_thresh_entry.bind('<Return>', self.set_new_thresh)

        return

    def build_islands_widgets(self):
        self.island_length_var = tk.StringVar()
        self.island_width_var = tk.StringVar()

        self.edit_centers_button = tk.Button(self.window, text="Edit Centers")
        self.edit_centers_button.grid(row=1, column=1)

        self.islands_instructions_label = tk.Label(
            self.window,
            text="""
            x: select island
            z/c: change select
            wasd: translate island
            e/r: rotate island
            """
        )
        self.islands_instructions_label.grid(row=2, column=1, rowspan=2)

        self.island_size_frame = tk.Frame(self.window)
        self.island_size_frame.grid(row=4, column=1, rowspan=2)

        self.island_length_label = tk.Label(self.island_size_frame, text="Island length: ")
        self.island_length_label.grid(row=1, column=1)

        self.island_length_entry = tk.Entry(self.island_size_frame, textvariable=self.island_length_var)
        self.island_length_entry.grid(row=1, column=2)

        self.island_width_label= tk.Label(self.island_size_frame, text="Island width: ")
        self.island_width_label.grid(row=2, column=1)

        self.island_width_entry = tk.Entry(self.island_size_frame, textvariable=self.island_width_var)
        self.island_width_entry.grid(row=2, column=2)

        self.islands_ok_button = tk.Button(self.window, text="Islands Ok", command=self.islands_ok)
        self.islands_ok_button.grid(row=7, column=1)

        self.islands_widgets = [
            self.edit_centers_button,
            self.islands_instructions_label,
            self.island_size_frame,
            self.island_length_label,
            self.island_length_entry,
            self.island_width_label,
            self.island_width_entry,
            self.islands_ok_button
        ]

        self.island_length_entry.bind('<Return>', self.change_island_length)
        self.island_width_entry.bind('<Return>', self.change_island_width)

        return

    def build_sigmas_widgets(self):
        self.color_map_var = tk.IntVar()
        self.outlines_var = tk.IntVar()

        self.edit_islands_button = tk.Button(self.window, text="Edit Islands")
        self.edit_islands_button.grid(row=1, column=1)

        self.color_map_box = tk.Checkbutton(self.window, text="Color map:",
                                            variable=self.color_map_var, command=self.color_checkbox)
        self.color_map_box.grid(row=2, column=1)
        self.color_map_box.select()

        self.outlines_box = tk.Checkbutton(self.window, text="Outlines:",
                                           variable=self.outlines_var, command=self.outlines_checkbox)
        self.outlines_box.grid(row=3, column=1)
        self.outlines_box.select()

        self.sigmas_instructions_label = tk.Label(
            self.window,
            text="""
            Click to select
            """
        )
        self.sigmas_instructions_label.grid(row=4, column=1)

        self.flip_button = tk.Button(self.window, text="Flip selected")
        self.flip_button.grid(row=5, column=1)

        self.save_results_button = tk.Button(self.window, text="Save results", command=self.save_data_to_file)
        self.save_results_button.grid(row=7, column=1)

        return

    def guides_ok(self):
        # Change mode
        self.mode = self.SET_CENTERS_MODE

        # Swap widgets
        for widget in self.guides_widgets:
            widget.grid_forget()
        self.build_centers_widgets()

        # Hide guides, draw all centers
        self.guide_points.hide_all()
        self.draw_centers()
        return

    def centers_ok(self):
        self.mode = self.SET_ISLANDS_MODE

        for widget in self.centers_widgets:
            widget.grid_forget()

        self.hide_centers()
        self.build_islands_widgets()
        self.create_aligned_islands()
        return

    def islands_ok(self):
        self.mode = self.SET_SIMGAS_MODE

        for widget in self.islands_widgets:
            widget.grid_forget()

        self.build_sigmas_widgets()
        self.draw_excised()
        self.draw_sigmas()
        self.draw_color_map()
        return

    def change_island_length(self, event=None):
        length = int(self.island_length_var.get())

        for island in self.islands:
            island.length = length

        self.delete_map()
        self.draw_map()
        return

    def change_island_width(self, event=None):
        width = int(self.island_width_var.get())

        for island in self.islands:
            island.width = width

        self.delete_map()
        self.draw_map()
        return

    def set_new_thresh(self, event=None):
        if self.mode == self.SET_CENTERS_MODE:  # TODO: replace with button state
            kernel = np.ones((5, 5), np.uint8)  # If the detection is bad, edit the kernel
            centers = MFM_Morph.get_centers(self.fp1, kernel, thresh=int(self.thresh_entry_var.get()))
            r = 3

            for shape in self.center_shapes:
                self.canvas.delete(shape)

            self.center_shapes = [  # TODO: These should really be combined in a class
                self.canvas.create_oval(center[0] - r, center[1] - r, center[0] + r, center[1] + r,
                                        outline='black', fill='red')
                for center in centers
            ]

            self.center_paths = [
                Path.circle(center=center, radius=r+1) for center in centers
            ]

            self.center_coords = centers

            self.current_centers_var.set("Current centers: {}".format(len(centers)))

        return

    def draw_centers(self, event=None):  # TODO: Combine this function with one above
        """
        Find and draw all detected shape centers.

        :return: none
        """

        if self.mode == self.SET_CENTERS_MODE:  # TODO: replace with button state
            kernel = np.ones((5, 5), np.uint8)  # If the detection is bad, edit the kernel
            centers = MFM_Morph.get_centers(self.fp1, kernel)
            r = 3

            if self.center_shapes:
                for shape in self.center_shapes:
                    self.canvas.delete(shape)

            self.center_shapes = [  # TODO: These should really be combined in a class
                self.canvas.create_oval(center[0] - r, center[1] - r, center[0] + r, center[1] + r,
                                        outline='black', fill='red')
                for center in centers
            ]

            self.center_paths = [
                Path.circle(center=center, radius=r+1) for center in centers
            ]

            self.center_coords = centers

            self.current_centers_var.set("Current centers: {}".format(len(centers)))
        return

    def delete_center(self):
        """
        Delete currently selected center.

        :return: none
        """
        if self.selected_center_ind is not None:  # TODO: add state variable instead
            self.canvas.delete(self.center_shapes[self.selected_center_ind])
            self.center_shapes.pop(self.selected_center_ind)
            self.center_paths.pop(self.selected_center_ind)
            self.center_coords = np.delete(self.center_coords, self.selected_center_ind, axis=0)
            self.selected_center_ind = None

            self.current_centers_var.set("Current centers: {}".format(len(self.center_coords)))
        return

    def create_aligned_islands(self):
        """
        Align the centers of the islands.

        :return: none
        """
        ideal_inds = np.genfromtxt(self.ideal_fp, delimiter=',')  # TODO: this doesn't belong here

        island_coords, theta_shift = MFM_Morph.align_centers(self.center_coords, ideal_inds[:, :2])

        self.islands = [
            Island(np.array([row[0][0], row[0][1]]), 20, 10, row[1][2] + theta_shift)
            for row in zip(island_coords, ideal_inds)  # TODO: length and width should be adjustable & not here
        ]

        self.center = np.mean(island_coords, axis=0)  # TODO: this is an odd place to define it
        self.draw_map()
        #self.draw_web()
        return

    def draw_map(self):
        """
        Draw outlines of islands onto canvas.

        :return: none
        """

        self.rec_objects = [self.canvas.create_polygon(*island.coords(), outline='red', fill='')
                            for island in self.islands]

        return

    def delete_map(self):
        for rec in self.rec_objects:
            self.canvas.delete(rec)
        return

    def draw_web(self):
        vertices = np.genfromtxt(r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\adjMatrices\single05.txt",
                                 delimiter=',')
        for vertex in vertices[:-1]:
            self.canvas.create_line(
                *self.islands[int(vertex[0])].center,
                *self.islands[int(vertex[1])].center,
                fill='yellow'
            )
            self.canvas.create_line(
                *self.islands[int(vertex[2])].center,
                *self.islands[int(vertex[3])].center,
                fill='yellow'
            )

    def draw_excised(self):
        """
        Calculate the coordinates of the rectangles that will be excised to represent the poles of each island.
        Plot the excised rectangles on the canvas.

        :return: none
        """
        scale_factor = 1/2  # TODO: magic number, make this a parameter

        self.inner_boxes = [
            Island(island.center, island.length * scale_factor, island.width, island.theta) for island in self.islands
        ]
        # TODO: all of these class variables should be defined in __init___
        self.excised_coords1 = [
            np.array([*inner.coords()[:4], *outer.coords()[2:4], *outer.coords()[:2]])
            for inner, outer in zip(self.inner_boxes, self.islands)
        ]

        self.excised_coords2 = [
            np.array([*inner.coords()[4:], *outer.coords()[6:], *outer.coords()[4:6]])
            for inner, outer in
            zip(self.inner_boxes, self.islands)
        ]

        # self.rec_excised1 = [
        #     self.canvas.create_polygon(
        #         *coords,
        #         outline='magenta', fill='')
        #     for coords in self.excised_coords1
        # ]
        # self.rec_excised2 = [
        #     self.canvas.create_polygon(
        #         *coords,
        #         outline='magenta', fill='')
        #     for coords in self.excised_coords2
        # ]
        return

    def draw_sigmas(self):
        """
        Draw the direction that the nanomagnets are pointing in on the canvas.

        :return: none
        """
        self.calculate_sigmas()

        # self.arrows = [
        #     self.canvas.create_line(
        #         *island.center,
        #         island.center[0]+np.cos(island.theta + np.pi * (island.sigma-1)/-2)*island.length/4,
        #         island.center[1]+np.sin(island.theta + np.pi * (island.sigma-1)/-2)*island.length/4,
        #         arrow=tk.LAST, fill='red'
        #     )
        #     for island in self.islands
        # ]
        return

    def calculate_sigmas(self):
        """
        Using the current island positions, calculate the sigma of each island.
        Excise islands using self.get_inds, decide sigma based on average of
        excised pixels.

        :return: none
        """
        # Read image as numpy array
        # IMPORTANT: PIL uses column-major, so this array has to be indexed as such.
        im_arr = np.asarray(self.im2)
        for i, island in enumerate(self.islands):
            # Get the coordinates of each excised pixel
            x1, y1 = self.get_inds(self.excised_coords1[i])
            x2, y2 = self.get_inds(self.excised_coords2[i])

            # Take the mean of the grayscale values of all the excised pixels
            av1 = np.mean(im_arr[y1, x1])  # indexing using column-major
            av2 = np.mean(im_arr[y2, x2])

            # Assign sigma based on which island is bigger
            if av1 > av2:
                island.sigma = 1
                # self.canvas.create_polygon(*self.excised_coords1[i], outline='green', fill='')
            elif av2 > av1:
                island.sigma = -1
                # self.canvas.create_polygon(*self.excised_coords2[i], outline='green', fill='')
            else:
                island.sigma = 0  # TODO: Add handling for edge case
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

        # Create a grid of every point in the rectangle that circumscribes the given rectangle
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

    def color_checkbox(self):
        if self.color_map_var.get():
            self.show_color_map()
        else:
            self.hide_color_map()
        return

    def outlines_checkbox(self):
        if self.outlines_var.get():
            self.show_recs()
        else:
            self.hide_recs()
        return

    def show_color_map(self):
        for item in self.color_islands:
            self.canvas.itemconfigure(item, state='normal')
        for item in self.little_black_arrows:
            self.canvas.itemconfigure(item, state='normal')
        return

    def hide_color_map(self):
        for item in self.color_islands:
            self.canvas.itemconfigure(item, state='hidden')
        for item in self.little_black_arrows:
            self.canvas.itemconfigure(item, state='hidden')
        return

    def show_recs(self):
        for item in self.rec_objects:
            self.canvas.itemconfigure(item, state='normal')
        return

    def hide_recs(self):
        for item in self.rec_objects:
            self.canvas.itemconfigure(item, state='hidden')
        return

    def show_centers(self):
        for item in self.center_shapes:
            self.canvas.itemconfigure(item, state='normal')
        return

    def hide_centers(self):
        for item in self.center_shapes:
            self.canvas.itemconfigure(item, state='hidden')
        return

    def draw_color_map(self):
        """
        Color the islands according to the angle at which they're magnetized. Create legend.

        :return: none
        """
        cmap = plt.get_cmap('hsv')
        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.color_islands = [
            self.canvas.create_polygon(
                *island.coords(),
                fill=mpl.colors.rgb2hex(scalar_map.to_rgba(
                    (island.theta + (lambda x: 0 if x == 1 else np.pi)(island.sigma)) % (2 * np.pi)
                ))
            )
            for island in self.islands
        ]

        self.little_black_arrows = [
            self.canvas.create_line(
                np.mean(self.inner_boxes[i].coords()[0:4:2]),
                np.mean(self.inner_boxes[i].coords()[1:4:2]),
                np.mean(self.inner_boxes[i].coords()[4:8:2]),
                np.mean(self.inner_boxes[i].coords()[5:8:2]),
                arrow=(lambda x: tk.FIRST if x == 1 else tk.LAST)(island.sigma),
                arrowshape=(4, 5, 2)
            )
            for i, island in enumerate(self.islands)
        ]

        # for theta in np.linspace(0, 2*np.pi, 200):
        #     self.color_legend.create_line(
        #         50, 50, 50+40*np.cos(theta), 50+40*np.sin(theta),
        #         fill=mpl.colors.rgb2hex(scalar_map.to_rgba(theta)),
        #         width=3,
        #     )
        # self.color_legend.create_oval(10, 10, 91, 91, fill='', outline='black', width=2)
        # self.color_legend.create_oval(45, 45, 55, 55, fill='black')
        return

    def image_1_button_click(self):
        """
        Make image 1 visible.

        :return: none
        """
        self.canvas.tag_lower(self.image2_canvas_object)
        return

    def image_2_button_click(self):
        """
        Make image 2 visible

        :return: none
        """
        self.canvas.tag_lower(self.image1_canvas_object)
        return

    def right_click(self, event):
        if self.mode == self.SET_GUIDES_MODE:
            self.guide_points.add_point([event.x, event.y])
        elif self.mode == self.SET_CENTERS_MODE:
            r = 3
            new_center = np.array([event.x, event.y])
            self.center_shapes.append(
                self.new_point(new_center)
            )
            self.center_paths.append(
                Path.circle(center=new_center, radius=4)
            )
            self.center_coords = np.vstack([self.center_coords, new_center])

            self.current_centers_var.set("Current centers: {}".format(len(self.center_coords)))
        return

    def new_point(self, center, r=3, fill='red', outline='black', **kwargs):
        """
        Draw a point on the canvas and return the object ID of the oval.

        :param center: array-like of length 2, [x, y] of point center
        :param **kwargs: will be passed to create_oval
        :return int: object ID of oval on canvas
        """
        return self.canvas.create_oval(
                    center[0] - r, center[1] - r, center[0] + r, center[1] + r,
                    fill=fill, outline=outline, **kwargs
                )

    def left_click(self, event):
        """
        If cursor is within a circle, select it.

        :param event: part of tkinter, automatically implemented through the bind mechanic
        :return: none
        """
        self.store_x = event.x  # TODO: These should be initialized somewhere else
        self.store_y = event.y

        if self.mode == self.SET_GUIDES_MODE:
            ind = self.guide_points.find_ind_by_loc([event.x, event.y])
            if ind is not None:
                self.guide_points.select_point(ind)
        elif self.mode == self.SET_CENTERS_MODE:
            r = 3  # TODO: SPOT
            for i, center_path in enumerate(self.center_paths):
                if center_path.contains_point((event.x, event.y)):
                    if self.selected_center_ind is not None:
                        self.canvas.delete(self.center_shapes[self.selected_center_ind])
                        coords = self.center_coords[self.selected_center_ind]
                        self.center_shapes[self.selected_center_ind] = self.canvas.create_oval(
                            coords[0] - r, coords[1] - r, coords[0] + r, coords[1] + r,
                            outline='black', fill='red'
                        )
                    self.selected_center_ind = i
                    self.canvas.delete(self.center_shapes[i])
                    coords = self.center_coords[i]
                    self.center_shapes[i] = self.canvas.create_oval(coords[0]-r, coords[1]-r, coords[0]+r, coords[1]+r,
                                                                    outline='black', fill='yellow')
        elif self.mode == self.SET_SIMGAS_MODE:
            for i, island in enumerate(self.islands):
                if (Path(island.coords().reshape((4, 2)))).contains_point((event.x, event.y)):
                    if self.selected_island_ind is not None:
                        self.canvas.delete(self.rec_objects[self.selected_island_ind])
                        if self.outlines_var.get():
                            state_var = 'normal'
                        else:
                            state_var = 'hidden'
                        self.rec_objects[self.selected_island_ind] = self.canvas.create_polygon(
                            *island.coords(),
                            outline='red', fill='', width=1, state=state_var
                        )
                    self.selected_island_ind = i
                    self.canvas.delete(self.rec_objects[i])
                    self.rec_objects[self.selected_island_ind] = self.canvas.create_polygon(
                        *island.coords(),
                        outline='yellow', fill='', width=3, state='normal'
                    )

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

        if self.mode == self.SET_GUIDES_MODE and self.guide_points.selected_ind is not None:
            self.guide_points.move_selected([delta_x, delta_y])
        elif self.mode == self.SET_CENTERS_MODE and self.selected_center_ind is not None:
            r = 3  # TODO: SPOT I AM BEGGING
            new_center = self.center_coords[self.selected_center_ind] + (delta_x, delta_y)
            self.canvas.delete(self.center_shapes[self.selected_center_ind])
            self.center_shapes[self.selected_center_ind] = self.canvas.create_oval(
                new_center[0] - r, new_center[1] - r, new_center[0] + r, new_center[1] + r,
                outline='black', fill='yellow'
            )
            self.center_paths[self.selected_center_ind] = Path.circle(center=new_center, radius=r+1)
            self.center_coords[self.selected_center_ind][0] = new_center[0]
            self.center_coords[self.selected_center_ind][1] = new_center[1]
        else:
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

        elif char == "l" and self.mode == self.SET_CENTERS_MODE and self.selected_center_ind is not None:
            self.delete_center()

        return

    def save_data_to_file(self):
        write_data = ""
        write_data += "{}".format(self.islands[0].sigma)
        for island in self.islands[1:]:
            write_data += "\n{}".format(island.sigma)

        with open(self.save_file, 'w+') as f:
            f.write(write_data)
        return
# Testing!
single10 = [
    r'C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_1.spm.png',
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00334_5.spm.png"
]
single9 = [
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00346_1.spm.png",
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00346_5.spm.png"
]

single6 = [
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00342_1.spm.png",
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00342_5.spm.png"
]

single5 = [
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00348_1.spm.png",
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00348_5.spm.png"
]

single_10_okay = [
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00332_1.spm.png",
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00332_5.spm.png"
]

single_10_blurry = [
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00331_cs_1.spm.png",
    r"C:\Users\sophi\Documents\School\SchifferLab\disclinations\Bingham\Disclination\MFMscanning.0_00331_cs_5.spm.png"
]

single_19 = [
    r"C:\Users\sh2547\Documents\data\Left MFM\MFMscanning.0_00521_1.spm.png",
    r"C:\Users\sh2547\Documents\data\Left MFM\MFMscanning.0_00521_5.spm.png"
]

# Testing!
# MFM_GUI(*single_19, 19, 'single', save_file=r'./savedresults/result.txt')
