import os

from MFM_GUI import MFM_GUI

# This just ensures that the current directory is in the same file as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

MFM_GUI(
    r"./example/image1.png",  # Filepath to the MFM where the island positions are clear
    r"./example/image2.png",  # Filepath to the MFM where the island magnetizations are clear
    n=19,  # n, or the number of islands on the side of the original "square"
    save_file=r"./example/results"  # Filepath to where results should be stored
)
