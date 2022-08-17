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

    guide_points = np.array([
        [166, 463], [531, 318],
        [531, 301], [217, 94],
        [202, 100], [152, 454],
        [294, 302], [297, 292], [283, 295]
    ])

    # Distort ideal map using guide points
    ideal_guide_ids = np.array([
        0, 16,
        50, 445,
        443, 17,
        280, 314, 312
    ])

    ideal_data = np.genfromtxt(ideal_fp, delimiter=',')
    ideal_centers = ideal_data[:,:2]

    sector_data = np.genfromtxt(r"C:\Users\sh2547\Documents\lab\sectordata\single19.csv", delimiter=',')

    A = {1: 0, 2: 2, 3: 4}
    B = {1: 1, 2: 3, 3: 5}
    C = {1: 6, 2: 7, 3: 8}

    colors_dict = {
        1: 'xkcd:cherry red',
        2: 'xkcd:purple',
        3: 'xkcd:yellow orange'
    }

    plt.imshow(img0)

    # for i, j in [[1, 3], [2, 1], [3, 2]]:
    #     primary = np.array([
    #         ideal_centers[ideal_guide_ids[A[i]]],
    #         ideal_centers[ideal_guide_ids[B[i]]],
    #         ideal_centers[ideal_guide_ids[C[i]]]
    #     ])
    #
    #     secondary = np.array([
    #         guide_points[A[j]],
    #         guide_points[B[j]],
    #         guide_points[C[j]]
    #     ])
    #
    #     sector_points = ideal_centers[sector_data[:, 1] == i, :]
    #
    #     transformation = affine_transformation(primary, secondary)
    #     transformed = transform_points(transformation, sector_points)
    #
    #     for center in transformed:
    #         plt.plot(*center, 'o', color=colors_dict[i], markersize=2)
    #
    # plt.show()

    primary = np.array([
        ideal_centers[ideal_guide_ids[0]],
        ideal_centers[ideal_guide_ids[2]],
        ideal_centers[ideal_guide_ids[4]]
    ])

    secondary = np.array([
        guide_points[4],
        guide_points[0],
        guide_points[2]
    ])

    t = affine_transformation(primary, secondary)
    p = transform_points(t, ideal_centers)

    plt.plot(p[:,0], p[:,1], 'o', color='xkcd:cherry red', markersize=2)
    plt.show()


    # fitted_points1 = fit_sector(1, 3, ideal_centers, ideal_guide_ids, guide_points, sector_data, img0)
    # fitted_points2 = fit_sector(2, 1, ideal_centers, ideal_guide_ids, guide_points, sector_data, img0)
    # fitted_points3 = fit_sector(3, 2, ideal_centers, ideal_guide_ids, guide_points, sector_data, img0)


    # for i, contour in enumerate(contours):
    #     if areas[i] <= area_max:
    #         if furthest_dist[i] <= furthest_dist_max:
    #             plt.fill(contour[:, 0], contour[:, 1], edgecolor='yellow', linewidth=0.5, color='yellow')
    #         else:
    #             plt.fill(contour[:, 0], contour[:, 1], edgecolor='orange', linewidth=0.5, color='orange')
    #     else:
    #         if furthest_dist[i] <= furthest_dist_max:
    #             plt.fill(contour[:, 0], contour[:, 1], edgecolor='red', linewidth=0.5, color='red')

    # for center in fitted_points1:
    #     plt.plot(*center, 'o', color='red', markersize=2)
    # for center in fitted_points2:
    #     plt.plot(*center, 'o', color='xkcd:azure', markersize=2)
    # for center in fitted_points3:
    #     plt.plot(*center, 'o', color='xkcd:green', markersize=2)

    real_middle = np.mean(guide_points[5:9], axis=0)

    # Use linear sum assignment for the confirmed points

    # Interpolate the gaps for remaining points

    # Profit?
    return


def extra_stuff_for_getting_info(ideal_data, ideal_guide_ids):
    ideal_centers = ideal_data[:,0:2]

    ideal_center = np.mean(ideal_centers[ideal_guide_ids[6:9], :], axis=0) + np.array([0.1, 0])
    pair1_center = np.mean(ideal_centers[ideal_guide_ids[[0, 5]], :], axis=0)
    pair2_center = np.mean(ideal_centers[ideal_guide_ids[1:3], :], axis=0)
    pair3_center = np.mean(ideal_centers[ideal_guide_ids[3:5], :], axis=0)

    m1, b1 = np.polyfit([ideal_center[0], pair1_center[0]], [ideal_center[1], pair1_center[1]], 1)
    m2, b2 = np.polyfit([ideal_center[0], pair2_center[0]], [ideal_center[1], pair2_center[1]], 1)
    m3, b3 = np.polyfit([ideal_center[0], pair3_center[0]], [ideal_center[1], pair3_center[1]], 1)

    guide = np.zeros(len(ideal_centers))
    guide[ideal_guide_ids] = np.array([1, 2, 1, 2, 1, 2, 3, 3, 3])
    sector = np.zeros(len(ideal_centers))

    for i, center in enumerate(ideal_centers):
        if m1*center[0] + b1 > center[1]:
            if m2*center[0] + b2 > center[1]:
                plt.plot(*center, 'bo', markersize=2)
                sector[i] = 1
            else:
                plt.plot(*center, 'ro', markersize=2)
                sector[i] = 2
        else:
            if m3*center[0] + b3 > center[1]:
                plt.plot(*center, 'ro', markersize=2)
                sector[i] = 2
            else:
                plt.plot(*center, 'go', markersize=2)
                sector[i] = 3
    plt.plot([ideal_center[0], pair1_center[0]], [ideal_center[1], pair1_center[1]], color='pink', linewidth=1)
    plt.plot([ideal_center[0], pair2_center[0]], [ideal_center[1], pair2_center[1]], color='pink', linewidth=1)
    plt.plot([ideal_center[0], pair3_center[0]], [ideal_center[1], pair3_center[1]], color='pink', linewidth=1)
    plt.xlim((-13, 10.5))
    plt.show()

    sector_data = np.vstack([guide, sector]).T
    np.savetxt(r"C:\Users\sh2547\Documents\lab\sectordata\single19.csv", sector_data, delimiter=",")

    if True:
        sector_data = np.genfromtxt(r"C:\Users\sh2547\Documents\lab\sectordata\single19.csv", delimiter=',')
        for i, center in enumerate(ideal_centers):
            color = {
                1: "bo",
                2: "ro",
                3: "go"
            }[sector_data[i, 1]]
            plt.plot(*center, color, markersize=2)

    return


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


def fit_sector(ideal, fit_to, ideal_centers, ideal_guide_ids, guide_points, sector_data, img):
    A = {1: 0, 2: 2, 3: 4}[ideal]
    B = {1: 1, 2: 3, 3: 5}[ideal]
    C = {1: 6, 2: 7, 3: 8}[ideal]
    I = {1: 0, 2: 2, 3: 4}[fit_to]
    J = {1: 1, 2: 3, 3: 5}[fit_to]
    K = {1: 6, 2: 7, 3: 8}[fit_to]

    points = ideal_centers[sector_data[:, 1] == ideal, :]
    # show_points(points, img)

    # Translate to origin
    delta_origin = -ideal_centers[ideal_guide_ids[A]]
    points += delta_origin
    # show_points(points, img)

    # Make horizontal
    theta1 = angle_between(ideal_centers[ideal_guide_ids[B]] - ideal_centers[ideal_guide_ids[A]], np.array([1, 0]))
    points = rotate(points, angle=theta1)
    # show_points(points, img)

    # Flip over x if needed
    if rotate(ideal_centers[ideal_guide_ids[C]],
              ideal_centers[ideal_guide_ids[A]],
              theta1)[1] > ideal_centers[ideal_guide_ids[A]][1]:
        bool1 = 1
    else:
        bool1 = 0
    theta3 = angle_between(guide_points[J] - guide_points[I], (1, 0))
    if rotate(guide_points[K], guide_points[I], theta3)[1] > guide_points[I][1]:
        bool2 = 1
    else:
        bool2 = 0
    if bool1 != bool2:
        points[:, 1] *= -1
    # show_points(points, img)

    # Dilate along base of triangle
    horizonal_factor = np.linalg.norm(
        guide_points[J] - guide_points[I]) / np.linalg.norm(
        ideal_centers[ideal_guide_ids[B]] - ideal_centers[ideal_guide_ids[A]])

    points[:, 0] *= horizonal_factor
    # show_points(points, img)

    # Dialate along height of triangle
    vertical_factor = np.linalg.norm(
        (guide_points[I] + guide_points[J]) / 2 - guide_points[K]) / np.linalg.norm(
        (ideal_centers[ideal_guide_ids[A]] + ideal_centers[ideal_guide_ids[B]]) / 2 - ideal_centers[ideal_guide_ids[C]])
    points[:, 1] *= vertical_factor
    # show_points(points, img)

    # Translate
    translate_delta = guide_points[I] - ideal_centers[ideal_guide_ids[A]]
    points = points - delta_origin + translate_delta
    # show_points(points, img)

    # Rotate
    theta2 = angle_between(np.array([1, 0]), guide_points[J] - guide_points[I])
    points = rotate(points, origin=guide_points[I], angle=theta2)
    # show_points(points, img)

    return points


def show_points(points, img):
    plt.imshow(img)

    for center in points:
        plt.plot(*center, 'ro', markersize=2)
    plt.show()
    return


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
    for i, point in enumerate(points):
        x, y = point
        points[i, 0] = x*m[0, 0] + y*m[1, 0] + m[2, 0]
        points[i, 1] = x * m[0, 1] + y * m[1, 1] + m[2, 1]
    return points

