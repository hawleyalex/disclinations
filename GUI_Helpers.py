import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from matplotlib import cm
from matplotlib.path import Path

from Island import Island


class PointList:
    def __init__(self, canvas, r=5, outline='black', fill='red', select_fill='yellow'):
        self.canvas = canvas
        self.r = r
        self.outline = outline
        self.fill = fill
        self.select_fill = select_fill

        self.object_ids = []
        self.paths = []
        self.coords = np.empty((0, 2))

        self.selected_ind = None
        self.state = 'normal'
        return

    def add_array(self, arr):
        for center in arr:
            self.object_ids.append(
                self.canvas.create_oval(
                    center[0] - self.r, center[1] - self.r, center[0] + self.r, center[1] + self.r,
                    outline='black', fill=self.fill
                )
            )
            self.paths.append(
                Path.circle(center=center, radius=self.r + 1)
            )
        self.coords = np.vstack([self.coords, arr])
        return

    def delete_all(self):
        for object_id in self.object_ids:
            self.canvas.delete(object_id)
        self.object_ids = []
        self.paths = []
        self.coords = np.empty((0, 2))
        return

    def add_point(self, center):
        self.object_ids.append(
            self.canvas.create_oval(
                center[0] - self.r, center[1] - self.r, center[0] + self.r, center[1] + self.r,
                outline='black', fill=self.fill
            )
        )
        self.paths.append(
            Path.circle(center=center, radius=self.r + 1)
        )
        self.coords = np.vstack([self.coords, np.array(center)])
        self.select_point(len(self.object_ids) - 1)
        return

    def delete_point(self, ind):
        self.canvas.delete(self.object_ids[self.selected_ind])
        self.object_ids.pop(ind)
        self.paths.pop(ind)
        self.coords = np.delete(self.coords, obj=ind, axis=0)
        return

    def select_point(self, ind):
        if self.selected_ind is not None and self.selected_ind < len(self.object_ids):
            self.canvas.itemconfig(self.object_ids[self.selected_ind], fill=self.fill)
        self.selected_ind = ind
        self.canvas.itemconfig(self.object_ids[self.selected_ind], fill=self.select_fill)
        return

    def find_ind_by_loc(self, point):
        for i, path in enumerate(self.paths):
            if path.contains_point(point):
                return i
        return None

    def move_selected(self, delta):
        new_center = self.coords[self.selected_ind] + np.array(delta)
        self.canvas.delete(self.object_ids[self.selected_ind])
        self.object_ids[self.selected_ind] = self.canvas.create_oval(
            new_center[0] - self.r, new_center[1] - self.r, new_center[0] + self.r, new_center[1] + self.r,
            outline=self.outline, fill=self.select_fill
        )
        self.paths[self.selected_ind] = Path.circle(center=new_center, radius=self.r + 1)
        self.coords[self.selected_ind][0] = new_center[0]
        self.coords[self.selected_ind][1] = new_center[1]

    def hide_all(self):
        for object_id in self.object_ids:
            self.canvas.itemconfigure(object_id, state='hidden')
        return

    def unhide_all(self):
        for object_id in self.object_ids:
            self.canvas.itemconfigure(object_id, state='normal')
        return


class IslandList:
    def __init__(self, canvas, islands):
        self.canvas = canvas
        self.islands = islands
        self.object_ids = []
        self.paths = []

        cmap = plt.get_cmap('hsv')
        norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

        self.fill = ''
        self.outline = 'red'
        self.selected_outline = 'yellow'
        self.lwd = 2
        self.selected_lwd = 2

        self.sigma_fill = lambda island: mpl.colors.rgb2hex(scalar_map.to_rgba(
            (island.theta + (lambda x: 0 if x == 1 else np.pi)(island.sigma)) % (2 * np.pi)))
        self.sigma_outline = ''
        self.sigma_selected_outline = 'yellow'
        self.sigma_selected_lwd = 4

        self.make_id_and_path()

        self.selected_ind = None
        self.inner_boxes = None
        self.excised_coords1 = None
        self.excised_coords2 = None
        self.excised_obj1 = None
        self.excised_obj2 = None
        self.sigma_ids = None
        self.little_black_arrows = None
        self.sigma_select_id = None
        return

    def make_id_and_path(self):
        self.object_ids = [self.canvas.create_polygon(*island.coords(), outline=self.outline, fill=self.fill,
                                                      width=self.lwd)
                           for island in self.islands]
        self.paths = [Path(island.coords().reshape((4, 2))) for island in self.islands]
        return

    def find_ind_by_loc(self, point):
        for i, path in enumerate(self.paths):
            if path.contains_point(point):
                return i
        return None

    def select_island(self, ind):
        if self.selected_ind is not None:
            self.canvas.itemconfig(self.object_ids[self.selected_ind], outline=self.outline)
        self.selected_ind = ind
        self.canvas.itemconfig(self.object_ids[self.selected_ind], outline=self.selected_outline)
        return

    def move_selected(self, delta):
        self.islands[self.selected_ind].center = self.islands[self.selected_ind].center + np.array(delta)
        self.canvas.delete(self.object_ids[self.selected_ind])
        self.object_ids[self.selected_ind] = self.canvas.create_polygon(*self.islands[self.selected_ind].coords(),
                                                                        outline=self.selected_outline,
                                                                        fill=self.fill,
                                                                        width=self.selected_lwd)
        self.paths[self.selected_ind] = Path(self.islands[self.selected_ind].coords().reshape((4, 2)))
        return

    def rotate_selected(self, theta):
        self.islands[self.selected_ind].theta = self.islands[self.selected_ind].theta + theta
        self.canvas.delete(self.object_ids[self.selected_ind])
        self.object_ids[self.selected_ind] = self.canvas.create_polygon(*self.islands[self.selected_ind].coords(),
                                                                        outline=self.selected_outline,
                                                                        fill=self.fill,
                                                                        width=self.selected_lwd)
        self.paths[self.selected_ind] = Path(self.islands[self.selected_ind].coords().reshape((4, 2)))
        return

    def calculate_excised(self, scale):
        self.inner_boxes = [
            Island(island.center, island.length * scale, island.width, island.theta) for island in self.islands
        ]

        self.excised_coords1 = [
            np.array([*inner.coords()[:4], *outer.coords()[2:4], *outer.coords()[:2]])
            for inner, outer in zip(self.inner_boxes, self.islands)
        ]

        self.excised_coords2 = [
            np.array([*inner.coords()[4:], *outer.coords()[6:], *outer.coords()[4:6]])
            for inner, outer in
            zip(self.inner_boxes, self.islands)
        ]

        self.excised_obj1 = [
            self.canvas.create_polygon(
                *coords,
                outline='magenta', fill='', state='hidden')
            for coords in self.excised_coords1
        ]
        self.excised_obj2 = [
            self.canvas.create_polygon(
                *coords,
                outline='magenta', fill='', state='hidden')
            for coords in self.excised_coords2
        ]
        return

    def show_excised(self):
        for object_id in self.excised_obj1 + self.excised_obj2:
            self.canvas.itemconfigure(object_id, state='normal')
        return

    def hide_excised(self):
        for object_id in self.excised_obj1 + self.excised_obj2:
            self.canvas.itemconfigure(object_id, state='hidden')
        return

    def show_islands(self):
        for object_id in self.object_ids:
            self.canvas.itemconfigure(object_id, state='normal')
        return

    def hide_islands(self):
        for object_id in self.object_ids:
            self.canvas.itemconfigure(object_id, state='hidden')
        return

    def add_sigmas(self):
        self.sigma_ids = [self.canvas.create_polygon(*island.coords(), outline=self.sigma_outline,
                                                     fill=self.sigma_fill(island)) for island in self.islands]

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
        return

    def hide_sigmas(self):
        for sigma_id in self.sigma_ids:
            self.canvas.itemconfigure(sigma_id, state='hidden')
        for arrow in self.little_black_arrows:
            self.canvas.itemconfigure(arrow, state='hidden')
        return

    def show_sigmas(self):
        for sigma_id in self.sigma_ids:
            self.canvas.itemconfigure(sigma_id, state='normal')
        for arrow in self.little_black_arrows:
            self.canvas.itemconfigure(arrow, state='normal')
        return

    def select_sigma(self, ind):
        if self.sigma_select_id is not None:
            self.canvas.delete(self.sigma_select_id)
        self.selected_ind = ind
        self.sigma_select_id = self.canvas.create_polygon(*self.islands[self.selected_ind].coords(),
                                                          outline=self.sigma_selected_outline, fill='',
                                                          width=self.sigma_selected_lwd)
        return

    def flip_selected_sigma(self):
        self.islands[self.selected_ind].sigma *= -1

        self.canvas.delete(self.sigma_ids[self.selected_ind])
        self.sigma_ids[self.selected_ind] = (lambda island: self.canvas.create_polygon(
            *island.coords(), outline=self.sigma_outline, fill=self.sigma_fill(island))
                                             )(self.islands[self.selected_ind])
        self.canvas.delete(self.little_black_arrows[self.selected_ind])
        self.little_black_arrows[self.selected_ind] = self.canvas.create_line(
                np.mean(self.inner_boxes[self.selected_ind].coords()[0:4:2]),
                np.mean(self.inner_boxes[self.selected_ind].coords()[1:4:2]),
                np.mean(self.inner_boxes[self.selected_ind].coords()[4:8:2]),
                np.mean(self.inner_boxes[self.selected_ind].coords()[5:8:2]),
                arrow=(lambda x: tk.FIRST if x == 1 else tk.LAST)(self.islands[self.selected_ind].sigma),
                arrowshape=(4, 5, 2)
            )
        self.canvas.tag_raise(self.sigma_select_id, self.sigma_ids[self.selected_ind])
        return

    def delete_all(self):
        for object_id in self.object_ids:
            self.canvas.delete(object_id)
        return

    def set_length(self, length):
        for island in self.islands:
            island.length = length
        self.delete_all()
        self.make_id_and_path()
        return

    def set_width(self, width):
        for island in self.islands:
            island.width = width
        self.delete_all()
        self.make_id_and_path()
        return
