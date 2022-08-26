# disclinations

Once you have cloned the repository, run `example.py` and follow along with the tutorial.

## Screen 1: Setting guides

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial1.PNG" width=50% />

This should be the first image that you see. It may look different on a different monitor and especially with a different operating system (I am using Windows 10). If some widgets aren't visible, try going fullscreen. If the window size is incorrect every time the program runs, you can change the default size in what is currently line 80 in the code (beginning with `self.window.geometry`). The proper syntax is `self.window.geometry("[width]x[length])`.

The `Im1` and `Im2` buttons in the upper right corner toggle which of the images is visible. The `Guides Ok` button is greyed out because there aren't any guide points on the screen yet. When the counter above it reaches `6/6`, the button will be enabled. The instructions on the left describe how to add, move, and delete guide points.

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial2.PNG" width=50% />

Once you have placed the guides, the screen should look like this. If the counter says that you have more than 6 guides but you only see 6, make sure that none of the guides are stacked on top of each other. Press `Guides Ok` to move to the next screen.

## Screen 2: Vetting centers

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial3.PNG" width=50% />

On this screen, you should see several red dots appear on the lattice. To have a successful analysis, it is import that all of these red dots are on island centers. The number of dots will intentionally be much lower than the number of islands.

Currently, the program is automatically choosing a threshold between 0-255 (black-white) in order to distinguish the background from the foreground. In this case, there are plenty of centered dots using the automatic threshold, but there may be instances where no dots appear on the screen at all. In this case, you can type a value between 0 and 255 into the "Set thresh" field and press enter. The higher the value, the lighter a point will have to be to be in the foreground. To reset the threshold to the automatic one first calculated, press `Reset thresh`.

If you see points that aren't on island centers, delete them or move them onto island centers. In this example, the points in the middle may be suspect since it's hard to see exactly where those centers are, but leaving them as they are won't strongly affect the results. However, the four points on the left and right edges of the screen will negatively affect the results. Makre sure to delete them before moving forward.

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial4.PNG" width=50% />

Once these points are deleted and every red point is on an island center, press `Centers Ok`.

## Screen 3: Adjusting islands

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial5.PNG" width=50% />

Each island can be moved and rotated individually by clicking on it and following the instructions in the upper left. The island length and width can be adjusted using the entries and pressing enter. (These measurements are in pixels.) Toggling between `Im1` and `Im2` here can help determine how large the islands should be to capture the black and white parts of the MFM.

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial6.PNG" width=50% />

Sometimes, especially with smaller lattices, the islands will be rotated in strange positions or won't be right on the centers. Pressing the `Try again` button will cycle through the six possible transformations that the program can create. To get the original configuration, press `Back` to go back a screen and then press `Centers Ok` again.

When the islands look reasonable aligned on `Im1` and `Im2`, press `Islands Ok`.

## Screen 4: Examining simgas

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial7.PNG" width=50% />

You can select and deselect the color map and outlines to hide or show them on the screen. You can also select an island and change its sigma if it appears to be incorrect.

After you press `Save results`, you should see a new folder in `./example` called `results`.

<img src="https://github.com/hawleyalex/disclinations/blob/new-detection/example/tutorial_photos/tutorial8.PNG" width=50% />


