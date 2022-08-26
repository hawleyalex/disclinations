import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk

from matplotlib import cm
from matplotlib.path import Path
from PIL import Image, ImageTk, ImageEnhance

from Island import Island
from GUI_Helpers import PointList, IslandList
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

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        plt.plot(0, 0)
        plt.clf()

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
        self.ideal_fp = r"./islandcoordinates/{}{:0>2d}.txt".format(self.ndisc, self.n_side)

        # Load images, convert to black and white
        self.im1 = Image.open(fp1).convert('L')
        self.im2 = Image.open(fp2).convert('L')

        enhancer = ImageEnhance.Contrast(self.im1)
        self.im_contrast = enhancer.enhance(3)
        im_x, im_y = self.im1.size
        im_x = int(im_x * 750 / im_y)
        im_y = 750

        self.newsize = (im_x, im_y)
        self.im1 = self.im1.resize(self.newsize)
        self.im2 = self.im2.resize(self.newsize)

        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

        # Create tk window
        self.window = tk.Tk()
        self.window.geometry(
            "{}x{}".format(im_x + 350, im_y + 50))

        self.tk_im1 = ImageTk.PhotoImage(self.im1)
        self.tk_im2 = ImageTk.PhotoImage(self.im2)

        # Create canvas widget and add images
        self.canvas = tk.Canvas(self.window, width=im_x * 1.5, height=im_y * 1.5)
        self.image2_canvas_object = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_im2)
        self.image1_canvas_object = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_im1)
        self.canvas.grid(row=2, column=2, rowspan=9)

        self.switch_image_frame = tk.Frame(self.window)
        self.switch_image_frame.grid(row=1, column=2)

        self.image_1_button = tk.Button(self.switch_image_frame, text="Im1", command=self.image_1_button_click)
        self.image_1_button.grid(row=0, column=0, sticky='ne')
        self.image_2_button = tk.Button(self.switch_image_frame, text="Im2", command=self.image_2_button_click)
        self.image_2_button.grid(row=0, column=1, sticky='nw')

        # Initialize lists
        self.guide_points = PointList(self.canvas)
        self.center_points = PointList(self.canvas)
        self.island_list = None

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
        self.selected_island_ind = None

        # Draw map of ideal island locations
        self.draw_centers()
        self.center_points.hide_all()

        # Set binds
        self.canvas.bind('<Button-1>', self.left_click)
        self.canvas.bind('<B1-Motion>', self.left_click_motion)
        self.canvas.bind('<Button-3>', self.right_click)
        self.window.bind('<Key>', self.key_press)

        # ideal_data = np.genfromtxt(self.ideal_fp, delimiter=',')
        # ideal_islands = [
        #     Island(data[0:2], 0.5, 0.25, data[2]) for data in ideal_data
        # ]
        # for i, island in enumerate(ideal_islands):
        #     plt.fill(island.coords()[::2], island.coords()[1::2], color='yellow')
        #     plt.text(*island.center, i, fontsize=5, ha='center', va='center')
        # plt.show()

        # Run GUI
        self.window.mainloop()

        return

    def build_guides_widgets(self):
        self.current_guides_var = tk.StringVar()
        self.current_guides_var.set("Current guides: {}/6".format(len(self.guide_points.object_ids)))

        self.guides_instructions_label = tk.Label(
            self.window,
            text="""
            Place points at each of the 
            six 'corner' islands
            
            Right click to place
            Left click to select and move
            Arrow keys for precision move
            Press L to delete selection
            """, anchor=tk.NW, justify='left'
        )
        self.guides_instructions_label.grid(row=2, column=1, rowspan=2)

        self.current_guides_label = tk.Label(self.window, textvariable=self.current_guides_var)
        self.current_guides_label.grid(row=4, column=1)

        self.guides_ok_button = tk.Button(self.window, text="Guides Ok", command=self.to_centers)
        self.guides_ok_button.grid(row=5, column=1)

        self.guides_widgets = [
            self.guides_instructions_label,
            self.current_guides_label,
            self.guides_ok_button
        ]

        if len(self.guide_points.object_ids) == 6:
            self.guides_ok_button["state"] = tk.NORMAL
        else:
            self.guides_ok_button["state"] = tk.DISABLED

        return

    def build_centers_widgets(self):
        self.thresh_entry_var = tk.StringVar()

        self.return_to_guides_button = tk.Button(self.window, text="Back", command=self.to_guides)
        self.return_to_guides_button.grid(row=1, column=1)

        self.thresh_frame = tk.Frame(self.window)
        self.thresh_frame.grid(row=2, column=1, rowspan=2)

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
        self.set_centers_instructions.grid(row=4, column=1)

        self.centers_ok_button = tk.Button(self.window, text="Centers Ok", command=self.to_islands)
        self.centers_ok_button.grid(row=5, column=1)

        self.centers_widgets = [
            self.return_to_guides_button,
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

        self.edit_centers_button = tk.Button(self.window, text="Back", command=self.to_centers)
        self.edit_centers_button.grid(row=1, column=1)

        self.islands_instructions_label = tk.Label(
            self.window,
            text="""
            click to select
            use arrow keys to move
            e/r: rotate island
            """
        )
        self.islands_instructions_label.grid(row=2, column=1, rowspan=2)

        self.island_size_frame = tk.Frame(self.window)
        self.island_size_frame.grid(row=4, column=1)

        self.island_length_label = tk.Label(self.island_size_frame, text="Island length: ")
        self.island_length_label.grid(row=1, column=1)

        self.island_length_entry = tk.Entry(self.island_size_frame, textvariable=self.island_length_var)
        self.island_length_entry.grid(row=1, column=2)

        self.island_width_label= tk.Label(self.island_size_frame, text="Island width: ")
        self.island_width_label.grid(row=2, column=1)

        self.island_width_entry = tk.Entry(self.island_size_frame, textvariable=self.island_width_var)
        self.island_width_entry.grid(row=2, column=2)

        self.try_again_button = tk.Button(self.window, text="Try again", command=self.change_transformation)
        self.try_again_button.grid(row=5, column=1)

        self.islands_ok_button = tk.Button(self.window, text="Islands Ok", command=self.to_sigmas)
        self.islands_ok_button.grid(row=6, column=1)

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

        self.edit_islands_button = tk.Button(self.window, text="Back", command=self.to_islands)
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

        self.flip_button = tk.Button(self.window, text="Flip selected", command=self.flip_button_click)
        self.flip_button.grid(row=5, column=1)

        self.save_results_button = tk.Button(self.window, text="Save results", command=self.save_data_to_file)
        self.save_results_button.grid(row=6, column=1)

        self.sigmas_widgets = [
            self.edit_islands_button,
            self.color_map_box,
            self.outlines_box,
            self.sigmas_instructions_label,
            self.flip_button,
            self.save_results_button
        ]

        return

    def to_guides(self):
        self.mode = self.SET_GUIDES_MODE

        for widget in self.centers_widgets:
            widget.grid_forget()
        self.build_guides_widgets()
        self.guide_points.unhide_all()
        self.center_points.hide_all()

        return

    def to_centers(self):
        if self.mode == self.SET_GUIDES_MODE:
            for widget in self.guides_widgets:
                widget.grid_forget()

            # Hide guides
            self.guide_points.hide_all()

        if self.mode == self.SET_ISLANDS_MODE:
            for widget in self.islands_widgets:
                widget.grid_forget()

            self.island_list.hide_islands()

        # Change mode
        self.center_points.unhide_all()
        self.mode = self.SET_CENTERS_MODE
        self.build_centers_widgets()
        return

    def to_islands(self):
        if self.mode == self.SET_CENTERS_MODE:
            for widget in self.centers_widgets:
                widget.grid_forget()

            self.center_points.hide_all()
            self.create_aligned_islands()
        elif self.mode == self.SET_SIMGAS_MODE:
            for widget in self.sigmas_widgets:
                widget.grid_forget()

            self.island_list.hide_sigmas()
            self.island_list.show_islands()

        self.mode = self.SET_ISLANDS_MODE

        self.build_islands_widgets()
        return

    def to_sigmas(self):
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

        self.island_list.set_length(length)
        return

    def change_island_width(self, event=None):
        width = int(self.island_width_var.get())

        self.island_list.set_width(width)
        return

    def set_new_thresh(self, event=None):
        kernel = np.ones((5, 5), np.uint8)  # If the detection is bad, edit the kernel
        centers = MFM_Morph.calculate_center_placement(self.fp1, self.newsize, kernel,
                                                       thresh=int(self.thresh_entry_var.get()))

        self.center_points.delete_all()
        self.center_points.add_array(centers)

        return

    def change_transformation(self):
        self.island_list.delete_all()
        self.create_aligned_islands(change_transform=True)
        return

    def flip_button_click(self):
        self.island_list.flip_selected_sigma()
        return

    def draw_centers(self, event=None):
        """
        Find and draw all detected shape centers.

        :return: none
        """

        kernel = np.ones((5, 5), np.uint8)  # If the detection is bad, edit the kernel
        centers = MFM_Morph.calculate_center_placement(self.fp1, self.newsize, kernel)

        self.center_points.add_array(centers)
        return

    def create_aligned_islands(self, change_transform=False):
        """
        Align the centers of the islands.

        :return: none
        """
        if change_transform:
            if self.transform_id == 5:
                set_i = 0
            else:
                set_i = self.transform_id + 1
        else:
            set_i = None
        self.islands, self.transform_id = MFM_Morph.calculate_final_islands(self.guide_points.coords,
                                                         self.ideal_fp, self.center_points.coords, self.n_side,
                                                         set_i=set_i)

        self.island_list = IslandList(self.canvas, self.islands)

        # self.draw_web()
        return

    def draw_web(self):
        vertices = np.genfromtxt(r".\vertices\single19.csv",  # TODO: Made nonlocal
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
        return

    def draw_excised(self):
        """
        Calculate the coordinates of the rectangles that will be excised to represent the poles of each island.
        Plot the excised rectangles on the canvas.

        :return: none
        """
        scale_factor = 1/2  # TODO: magic number, make this a parameter

        self.island_list.calculate_excised(scale_factor)

        return

    def draw_sigmas(self):
        """
        Draw the direction that the nanomagnets are pointing in on the canvas.

        :return: none
        """
        self.calculate_sigmas()
        self.island_list.add_sigmas()
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
        for i, island in enumerate(self.island_list.islands):
            # Get the coordinates of each excised pixel
            x1, y1 = self.get_inds(self.island_list.excised_coords1[i])
            x2, y2 = self.get_inds(self.island_list.excised_coords2[i])

            # Take the mean of the grayscale values of all the excised pixels
            av1 = np.mean(im_arr[y1, x1])  # indexing using column-major
            av2 = np.mean(im_arr[y2, x2])

            # Assign sigma based on which island is bigger
            if av1 > av2:
                island.sigma = 1
            elif av2 > av1:
                island.sigma = -1
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
        self.island_list.show_sigmas()
        return

    def hide_color_map(self):
        self.island_list.hide_sigmas()
        return

    def show_recs(self):
        self.island_list.show_islands()
        return

    def hide_recs(self):
        self.island_list.hide_islands()
        return

    def draw_color_map(self):
        """
        Color the islands according to the angle at which they're magnetized. Create legend.

        :return: none
        """

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
            n = len(self.guide_points.object_ids)
            self.current_guides_var.set("Current guides: {}/6".format(n))
            if n == 6:
                self.guides_ok_button["state"] = tk.NORMAL
            else:
                self.guides_ok_button["state"] = tk.DISABLED
        elif self.mode == self.SET_CENTERS_MODE:
            self.center_points.add_point([event.x, event.y])
        return

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
            ind = self.center_points.find_ind_by_loc([event.x, event.y])
            if ind is not None:
                self.center_points.select_point(ind)
        elif self.mode == self.SET_ISLANDS_MODE:
            ind = self.island_list.find_ind_by_loc([event.x, event.y])
            if ind is not None:
                self.island_list.select_island(ind)
        elif self.mode == self.SET_SIMGAS_MODE:
            ind = self.island_list.find_ind_by_loc([event.x, event.y])
            if ind is not None:
                self.island_list.select_sigma(ind)
        return

    def left_click_motion(self, event):
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
        elif self.mode == self.SET_CENTERS_MODE and self.center_points.selected_ind is not None:
            self.center_points.move_selected([delta_x, delta_y])
        elif self.mode == self.SET_ISLANDS_MODE and self.island_list.select_island is not None:
            self.island_list.move_selected([delta_x, delta_y])
        return

    def arrow_key(self, delta):
        if self.mode == self.SET_GUIDES_MODE:
            self.guide_points.move_selected(delta)
        elif self.mode == self.SET_CENTERS_MODE:
            self.center_points.move_selected(delta)
        elif self.mode == self.SET_ISLANDS_MODE:
            self.island_list.move_selected(delta)
        return

    def key_press(self, event):
        """
        Makes GUI responsive to keyboard.

        :param event: tkinter variable, not necessary
        :return: none
        """
        char = event.keysym

        # Move points or islands
        if char == "Left":
            self.arrow_key([-1, 0])
        elif char == "Right":
            self.arrow_key([1, 0])
        elif char == "Up":
            self.arrow_key([0, -1])
        elif char == "Down":
            self.arrow_key([0, 1])

        # Rotate island
        elif char == "r":
            if self.mode == self.SET_ISLANDS_MODE and self.island_list.selected_ind is not None:
                self.island_list.rotate_selected(np.pi / 64)
        elif char == "e":
            if self.mode == self.SET_ISLANDS_MODE and self.island_list.selected_ind is not None:
                self.island_list.rotate_selected(-np.pi / 64)

        # Delete point
        elif char == "l":
            if self.mode == self.SET_GUIDES_MODE and self.guide_points.selected_ind is not None:
                self.guide_points.delete_point(self.guide_points.selected_ind)
                n = len(self.guide_points.object_ids)
                self.current_guides_var.set("Current guides: {}/6".format(n))
                if n == 6:
                    self.guides_ok_button["state"] = tk.NORMAL
                else:
                    self.guides_ok_button["state"] = tk.DISABLED
            elif self.mode == self.SET_CENTERS_MODE and self.center_points.selected_ind is not None:
                self.center_points.delete_point(self.center_points.selected_ind)

        return

    def save_data_to_file(self):
        if not os.path.exists(self.save_file):
            os.mkdir(self.save_file)

        write_data = "id,x,y,theta,sigma"
        for i, island in enumerate(self.island_list.islands):
            write_data += "\n{},{},{},{},{}".format(i, island.center[0], island.center[1], island.theta, island.sigma)

        with open(self.save_file + r"\data.csv", 'w+') as f:
            f.write(write_data)

        img = np.asarray(self.im2)
        plt.imshow(img, cmap='gray', vmin=0, vmax=225)

        cmap = plt.get_cmap('hsv')
        norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

        sigma_fill = lambda island: mpl.colors.rgb2hex(scalar_map.to_rgba(
            (island.theta + (lambda x: 0 if x == 1 else np.pi)(island.sigma)) % (2 * np.pi)))

        for island, inner_box in zip(self.island_list.islands, self.island_list.inner_boxes):
            plt.fill(island.coords()[::2], island.coords()[1::2], fc=sigma_fill(island), ec=sigma_fill(island))
            if island.sigma == -1:
                plt.arrow(
                    np.mean(inner_box.coords()[0:4:2]),
                    np.mean(inner_box.coords()[1:4:2]),
                    dx=-inner_box.length * np.cos(inner_box.theta),
                    dy=-inner_box.length * np.sin(inner_box.theta),
                    head_width=2.5,
                    head_length=2.5,
                    lw=1,
                    length_includes_head=True,
                )
            else:
                plt.arrow(
                    np.mean(inner_box.coords()[4:8:2]),
                    np.mean(inner_box.coords()[5:8:2]),
                    dx=inner_box.length * np.cos(inner_box.theta),
                    dy=inner_box.length * np.sin(inner_box.theta),
                    head_width=2.5,
                    head_length=2.5,
                    lw=1,
                    length_includes_head=True,
                )
        plt.savefig(self.save_file + r"\spin_directions.png")

        plt.clf()

        img = np.asarray(self.im1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=225)

        for i, island in enumerate(self.island_list.islands):
            plt.text(*island.center, i, size=2.5, color="white", ha='center', va='center',
                     path_effects=[pe.withStroke(linewidth=1, foreground="red")])
        plt.savefig(self.save_file + r"\island_numbers.png")

        return
