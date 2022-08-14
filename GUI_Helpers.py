import numpy as np

from matplotlib.path import Path


class PointList:
    def __init__(self, canvas, r=3, outline='black', fill='red', select_fill='yellow'):
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
        return

    def delete_point(self, ind):
        self.canvas.delete(self.object_ids[self.selected_ind])
        self.object_ids.pop(ind)
        self.paths.pop(ind)
        self.coords = np.delete(self.coords, obj=ind, axis=0)
        return

    def select_point(self, ind):
        if self.selected_ind is not None:
            self.canvas.delete(self.object_ids[self.selected_ind])
            center = self.coords[self.selected_ind]
            self.object_ids[self.selected_ind] = self.canvas.create_oval(
                center[0] - self.r, center[1] - self.r, center[0] + self.r, center[1] + self.r,
                outline=self.outline, fill=self.fill
            )
        self.selected_ind = ind
        self.canvas.delete(self.object_ids[ind])
        center = self.coords[ind]
        self.object_ids[self.selected_ind] = self.canvas.create_oval(
            center[0] - self.r, center[1] - self.r, center[0] + self.r, center[1] + self.r,
            outline=self.outline, fill=self.select_fill
        )
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
    def __init__(self, canvas, islands, outline='red', fill='', selected_outline='yellow', selected_lwd=1):
        self.canvas = canvas
        self.islands = islands
        self.object_ids = []
        self.paths = []

        for island in islands:
            self.make_id_and_path(island)

        self.outline = outline
        self.fill = fill
        self.selected_outline = selected_outline
        self.selected_lwd = selected_lwd

        self.selected_ind = None
        return

    def make_id_and_path(self, island):
        return

    def find_island_by_loc(self, point):
        return

    def select_island(self, ind):
        if self.selected_ind is not None:
            pass
        self.selected_ind = ind
        return

    def translate_selected(self, delta):
        return

    def rotate_selected(self, theta):
        return

    def calculate_excised(self, scale):
        return

    def show_excised(self):
        return

    def hide_excised(self):
        return

    def show_islands(self):
        return

    def hide_islands(self):
        return
