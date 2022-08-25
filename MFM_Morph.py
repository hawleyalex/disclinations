import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.path import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from Island import Island


def calculate_center_placement(img_fp, dim, kernel, thresh=None):
    """
    @param: img_fp, str, path to MFM image
    @param: kernel, numpy array, kernel for image morphology
    @param: thresh, int, 0-255 threshold for image morphology

    @return: numpy array with shape (N, 2), centers of likely island shapes
    """
    # All the morphology stuff, get the centers
    img0 = cv2.imread(img_fp)
    img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dim)

    # Make every pixel black or white depending on threshold (calculated by cv2.THRESH_OTSU)
    # Basically increase contrast to maximum possible

    if thresh is None:
        ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, img_thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # cv2.imshow("thresh", img_thresh)
    # cv2.waitKey()

    # Remove any white shape that is smaller than kernel
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("opening", opening)
    # cv2.waitKey()

    # Get each contour that surrounds a white shape
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_all_pts, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour.reshape(-1, 2) for contour in contours]

    all_centers = [np.mean(contour.reshape(-1, 2), axis=0) for contour in contours_all_pts]

    # Get area of each contour
    paths = [Path(contour) for contour in contours]
    areas = []
    for i, contour in enumerate(contours):
        min_x, max_x = min(contour[:,0]), max(contour[:,0])
        min_y, max_y = min(contour[:,1]), max(contour[:,1])

        x, y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        grid = paths[i].contains_points(points)
        mask = grid.reshape(max_y - min_y + 1, max_x - min_x + 1)  # make bool mask for points inside the given rect
        areas.append(np.sum(mask))

    # Get distance of two furthest points
    furthest_dist = []
    for contour in contours:
        max_dist2 = 0
        for i in range(len(contour)):
            for j in range(i):
                dist2 = np.sum((contour[i] - contour[j])**2)
                if dist2 > max_dist2:
                    max_dist2 = dist2
        furthest_dist.append(np.sqrt(max_dist2))

    # plt.boxplot(furthest_dist)
    # plt.show()
    # plt.clf()

    area_max = np.quantile(areas, 0.75)
    area_min = np.quantile(areas, 0.25)
    furthest_dist_max = np.quantile(furthest_dist, 0.75)

    index_array = np.array((areas <= area_max) & (furthest_dist <= furthest_dist_max) & (areas >= area_min))

    centers = np.array(all_centers)[index_array]

    return centers


def calculate_final_islands(guide_points, ideal_fp, centers):
    """
    @param: guide_points, numpy array, should have shape (6, 2)
    @param: ideal_fp, str, path to center of island locations
    @param: centers, numpy array, shape (N, 2), experimental centers o fislands

    @return: list of Island
    """
    guides = np.zeros((3, 2))
    middle = np.mean(guide_points, axis=0)
    pairs = find_pairs(guide_points)
    for i, pair in enumerate(pairs):
        if angle_between(guide_points[pair[0]] - middle, guide_points[pair[1]] - middle) > 0:
            guides[i] = guide_points[pair[0]]
        else:
            guides[i] = guide_points[pair[1]]
            guides[i] = guide_points[pair[1]]

    if angle_between(guides[0] - middle, guides[1] - middle) < 0:
        save = guides[1].copy()
        guides[1] = guides[2]
        guides[2] = save

    # Distort ideal map using guide points
    ideal_guide_ids = np.array([
        0, 16,
        50, 445,
        443, 17,
        280, 314, 312
    ])

    ideal_data = np.genfromtxt(ideal_fp, delimiter=',')
    ideal_centers = ideal_data[:,:2]

    A = {1: 0, 2: 2, 3: 4}
    B = {1: 1, 2: 3, 3: 5}
    C = {1: 6, 2: 7, 3: 8}

    #

    primary1 = np.array([
        ideal_centers[ideal_guide_ids[1]],
        ideal_centers[ideal_guide_ids[3]],
        ideal_centers[ideal_guide_ids[5]]
    ])

    primary2 = np.array([
        ideal_centers[ideal_guide_ids[0]],
        ideal_centers[ideal_guide_ids[2]],
        ideal_centers[ideal_guide_ids[4]]
    ])

    secondary1 = np.array([
        guides[0],
        guides[1],
        guides[2]
    ])
    secondary2 = np.array([
        guides[2],
        guides[0],
        guides[1]
    ])
    secondary3 = np.array([
        guides[1],
        guides[2],
        guides[0]
    ])
    secondary4 = np.array([
        guides[2],
        guides[1],
        guides[0]
    ])
    secondary5 = np.array([
        guides[0],
        guides[2],
        guides[1]
    ])
    secondary6 = np.array([
        guides[1],
        guides[0],
        guides[2]
    ])

    t1 = affine_transformation(primary1, secondary1)
    t2 = affine_transformation(primary1, secondary2)
    t3 = affine_transformation(primary1, secondary3)
    t4 = affine_transformation(primary2, secondary4)
    t5 = affine_transformation(primary2, secondary5)
    t6 = affine_transformation(primary2, secondary6)
    p1 = transform_points(t1, ideal_centers)
    p2 = transform_points(t2, ideal_centers)
    p3 = transform_points(t3, ideal_centers)
    p4 = transform_points(t4, ideal_centers)
    p5 = transform_points(t5, ideal_centers)
    p6 = transform_points(t6, ideal_centers)

    # plt.imshow(img0)
    # plt.plot(centers[:,0], centers[:,1], 'ro', markersize=1)
    # print(contours)
    # print(index_array)
    # for i in range(len(index_array)):
    #     if index_array[i]:
    #         plt.plot(contours[i][:,0], contours[i][:,1], color='yellow', linewidth=1)
    #
    # plt.show()

    min_cost = float('inf')
    min_i = None
    for ind, p in enumerate([p1, p2, p3, p4, p5, p6]):
        cost_matrix = np.zeros((len(p), len(centers)))
        for i in range(len(p)):
            for j in range(len(centers)):
                # cost is square of distance
                cost_matrix[i, j] = np.sum((p[i] - centers[j]) ** 2)

        # Use linear sum assignment to move the ideal centers to the closest mfm centers
        ideal_inds, mfm_inds = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[ideal_inds, mfm_inds].sum()

        if cost < min_cost:
            min_cost = cost
            min_i = ind

    p = [p1, p2, p3, p4, p5, p6][min_i]
    t = [t1, t2, t3, t4, t5, t6][min_i]

    ideal_islands = [
        Island(point, length=0.4, width=0.2, theta=ideal_data[i, 2]) for i, point in enumerate(ideal_centers)
    ]

    if min_i in [3, 4, 5]:
        flipped = True
    else:
        flipped = False
    for island in ideal_islands:
        reshaped_coords = island.coords().reshape((4, 2))
        transformed = transform_points(t, reshaped_coords)
        island.enter_coords(np.squeeze(transformed.reshape((8, 1))), flipped=flipped)

    return ideal_islands


def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def angle_between(a, b):

    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    if a[0] * b[1] - a[1] * b[0] < 0:
        rad = -rad
    return rad


def affine_transformation(primary, secondary):
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    transform = lambda x: unpad(np.dot(pad(x), A))

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return A


def transform_points(m, points):
    trans_points = np.zeros(points.shape)
    for i, point in enumerate(points):
        x, y = point
        trans_points[i, 0] = x * m[0, 0] + y * m[1, 0] + m[2, 0]
        trans_points[i, 1] = x * m[0, 1] + y * m[1, 1] + m[2, 1]
    return trans_points


def find_pairs(guide_points):
    dmat = distance_matrix(guide_points, guide_points)

    pairs_found = 0
    pairs = np.full((3, 2), -1)
    for i, row in enumerate(dmat):
        match = np.argpartition(row, 1)[1]
        if pairs_found:
            new = True
            for found_pair in pairs[:pairs_found,]:
                if found_pair[0] == match:
                    new = False
            if new:
                pairs[pairs_found] = np.array([i, match])
                pairs_found += 1
                if pairs_found == 3:
                    return pairs
        else:
            pairs[pairs_found] = np.array([i, match])
            pairs_found += 1
    return None
