import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.path import Path
from scipy.optimize import linear_sum_assignment


def get_centers(img_fp, kernel, thresh=None):
    """
    Return the centers of each shape in the foreground of the image.
    The centers currently aren't accurate, but they're guaranteed to be within the shape.

    :param img_fp: string, the filepath to an image with lighter shapes in the foreground
    :param kernel: numpy array, a shape that will be used for image morphology. Change this if shape detection is bad.
    :param thresh: int, number from 0-255 that changes the detection threshold for the image
    :return: centers, numpy array size (# of shapes, 2)
    """
    # Read the image into cv2 in grayscale
    img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)

    # Make every pixel black or white depending on threshold (calculated by cv2.THRESH_OTSU)
    # Basically increase contrast to maximum possible

    if thresh is None:
        ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, img_thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # Remove any white shape that is smaller than kernel
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # Get each contour that surrounds a white shape
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Use a pretty inaccurate method of calculating the center of each white shape
    # (It's guaranteed to be inside the shape, but it will be biased towards the "ridgier" side)
    # This could be mitigated by (1) not using a chain approximation or (2) making another algorithm
    # that would simplify the shape.
    centers = np.array([
        np.mean(contour, axis=0)[0] for contour in contours
    ])

    return centers


def align_centers(mfm_centers, ideal_centers):
    num_centers = len(ideal_centers)

    # Approximate the center of the mfm by averaging the coordinates of all the islands
    approx_center = np.mean(mfm_centers, axis=0)

    # Find the three closest points to the approximate center, which should be the three central islands
    dist_2 = np.sum((mfm_centers - approx_center) ** 2, axis=1)
    three_closest_inds = np.argpartition(dist_2, 3)[:3]
    three_closest = mfm_centers[three_closest_inds]

    # Approximate the center again by taking the average of the coordinates of the three center points
    better_center = np.mean(three_closest, axis=0)

    # Find the four furthest points from the center, two of which should be points on the
    # same "tip" of the dorito
    dist_2_better_center = np.sum((mfm_centers - better_center) ** 2, axis=1)
    four_furthest_inds = np.argpartition(dist_2_better_center, -4)[-4:]

    # Find which two points are on the same tip of the dorito
    # This is not optimized well, but it's a short one-time operation so it shouldn't matter
    min_pair = None
    min_dist = 1000000
    for i in range(4):
        for j in range(4):
            if i < j:
                c1 = mfm_centers[four_furthest_inds[i]]
                c2 = mfm_centers[four_furthest_inds[j]]
                new_dist = np.sqrt(np.sum((c1 - c2) ** 2))
                if new_dist < min_dist:
                    min_dist = new_dist
                    min_pair = (i, j)

    # Determine which of the two has a greater angle from the horizontal
    coords1 = mfm_centers[four_furthest_inds[min_pair[0]]] - better_center
    theta1 = np.arctan(coords1[1] / coords1[0])
    coords2 = mfm_centers[four_furthest_inds[min_pair[1]]] - better_center
    theta2 = np.arctan(coords2[1] / coords2[0])

    # Select the center that is on the left/right TODO
    if theta1 > theta2:
        furthest_ind = four_furthest_inds[min_pair[0]]
    else:
        furthest_ind = four_furthest_inds[min_pair[1]]

    furthest_distance = np.sqrt(np.sum((better_center - mfm_centers[furthest_ind]) ** 2))

    # Here, we use the fact that we know that ideal_centers[0] is a specific dorito tip point
    ideal_center = np.mean(ideal_centers, axis=0)
    furthest_dist_ideal = np.sqrt(np.sum((ideal_center - ideal_centers[0]) ** 2))

    # Dilate and shift the ideal map so that the two points can overlap
    dilation_factor = furthest_distance / furthest_dist_ideal
    ideal_centers_dilated = ideal_centers * dilation_factor
    ideal_centers_shifted = ideal_centers_dilated + better_center

    # Calculate the angle between the known ideal point and the actual dorito tip
    ideal_far_point_coords = ideal_centers_shifted[0] - better_center
    ideal_far_point_theta = np.arctan(ideal_far_point_coords[1] / ideal_far_point_coords[0])
    if ideal_far_point_coords[0] < 0:  # Deal with arctan
        ideal_far_point_theta += np.pi

    mfm_far_point_coords = mfm_centers[furthest_ind] - better_center
    mfm_far_point_theta = np.arctan(mfm_far_point_coords[1] / mfm_far_point_coords[0])
    if mfm_far_point_coords[0] < 0:
        mfm_far_point_theta += np.pi
    theta_shift = mfm_far_point_theta - ideal_far_point_theta

    # Shift each center by theta_shift
    for i, center in enumerate(ideal_centers_shifted):
        cos_th = np.cos(theta_shift)
        sin_th = np.sin(theta_shift)
        delta_x = center[0] - better_center[0]
        delta_y = center[1] - better_center[1]
        ideal_centers_shifted[i, 0] = better_center[0] + cos_th * delta_x - sin_th * delta_y
        ideal_centers_shifted[i, 1] = better_center[1] + sin_th * delta_x + cos_th * delta_y

    # Create a matrix of the distances between each pair of ideal and mfm points
    cost_matrix = np.zeros((num_centers, num_centers))
    for i in range(num_centers):
        for j in range(num_centers):
            # cost is square of distance
            cost_matrix[i, j] = np.sum((mfm_centers[i] - ideal_centers_shifted[j]) ** 2)

    # Use linear sum assignment to move the ideal centers to the closest mfm centers
    mfm_inds, ideal_inds = linear_sum_assignment(cost_matrix)

    sorted_island_data = np.zeros((num_centers, 2))
    for i, ind in enumerate(ideal_inds):
        sorted_island_data[ind] = mfm_centers[i]

    return sorted_island_data, theta_shift


def calculate_center_placement(img_fp, ideal_fp, kernel, thresh=None):
    # All the morphology stuff, get the centers
    img0 = cv2.imread(img_fp)
    img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)

    # Make every pixel black or white depending on threshold (calculated by cv2.THRESH_OTSU)
    # Basically increase contrast to maximum possible

    if thresh is None:
        ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, img_thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # Remove any white shape that is smaller than kernel
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # Get each contour that surrounds a white shape
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_all_pts, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour.reshape(-1, 2) for contour in contours]

    centers = [np.mean(contour.reshape(-1, 2), axis=0) for contour in contours_all_pts]

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

    area_max = np.quantile(areas, 0.80)
    furthest_dist_max = np.quantile(furthest_dist, 0.80)

    plt.imshow(img0)
    for i, contour in enumerate(contours):
        if areas[i] <= area_max:
            if furthest_dist[i] <= furthest_dist_max:
                plt.fill(contour[:,0], contour[:,1], edgecolor='yellow', linewidth=0.5, color='yellow')
            else:
                plt.fill(contour[:, 0], contour[:, 1], edgecolor='orange', linewidth=0.5, color='orange')
        else:
            if furthest_dist[i] <= furthest_dist_max:
                plt.fill(contour[:, 0], contour[:, 1], edgecolor='red', linewidth=0.5, color='red')
    #plt.show()


    guide_points = np.array([
        [166, 463], [531, 318],
        [531, 301], [217, 94],
        [202, 100], [152, 454],
        [294, 302], [297, 292], [283, 295]
    ])

    # Distort ideal map using guide points
    ideal_data = np.genfromtxt(ideal_fp, delimiter=',')
    ideal_centers = ideal_data[:,:2]
    ideal_middle = np.mean(ideal_centers, axis=0)

    dist2 = np.sum((ideal_centers - ideal_middle)**2, axis=1)
    inds = np.argsort(dist2)
    ideal_inds = np.sort(inds)




    plt.clf()
    for center in ideal_centers[inds[3:-6],:]:
        plt.plot(*center, 'ro')
    for center in ideal_centers[inds[-6:],:]:
        plt.plot(*center, 'bo')
    for center in ideal_centers[inds[:3],:]:
        plt.plot(*center, 'bo')
    plt.xlim((-13, 10.5))
    plt.show()

    real_middle = np.mean(guide_points[5:9], axis=0)

    # Use linear sum assignment for the confirmed points

    # Interpolate the gaps for remaining points

    # Profit?
    return
