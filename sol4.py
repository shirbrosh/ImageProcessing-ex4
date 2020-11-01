import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy.ndimage.interpolation import map_coordinates
import shutil
from imageio import imwrite
import sol4_utils
from scipy.ndimage.filters import convolve
import random

DERIVATIVE_FILTER = np.array([[1, 0, -1]])
KERNEL_SIZE = 3
POWER = 2
K = 0.04
TRANSFORM_COOR_LAYERS = 0.25
COL = 1
PYR_FIRST_LEVEL =0
PYR_3RD_LEVEL = 2
SECOND_MAX = -2
IDENTITY_MATRIX = 3


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of
    the ith corner points.
    """
    # calculate derivative
    Ix = convolve(im, DERIVATIVE_FILTER)
    Iy = convolve(im, DERIVATIVE_FILTER.T)

    # create matrix M
    M = np.array([
        [sol4_utils.blur_spatial(Ix * Ix, KERNEL_SIZE),
         sol4_utils.blur_spatial(Ix * Iy, KERNEL_SIZE)],
        [sol4_utils.blur_spatial(Iy * Ix, KERNEL_SIZE),
         sol4_utils.blur_spatial(Iy * Iy, KERNEL_SIZE)]])

    det_M = (M[0, 0] * M[1, 1]) - (M[0, 1] * M[1, 0])
    trace_M = M[0, 0] + M[1, 1]
    R = det_M - K * (np.power(trace_M, POWER))

    bool_im = non_maximum_suppression(R.T)
    corners = np.where(bool_im)
    corners_arr = np.stack(corners, axis=1)
    return corners_arr


def sample_descriptor_one_point(pos, im, desc_rad):
    """
    This function operates samples descriptor for one point
    :param pos: the point to sample
    :param im: the image from which the points was taken
    :param desc_rad: "Radius" of descriptors to compute.
    :return: a 2D array with shape (K,K) containing the descriptor for the given point
    """
    x_cor = np.arange(pos[0] - desc_rad, pos[0] + desc_rad + 1)
    y_cor = np.arange(pos[1] - desc_rad, pos[1] + desc_rad + 1)
    yy, xx = np.meshgrid(x_cor, y_cor)
    window = map_coordinates(im, [xx, yy], order=1, prefilter=False)
    mean_window = window - np.mean(window)
    if np.linalg.norm(mean_window) == 0:
        return mean_window
    final_window = mean_window / np.linalg.norm(mean_window)
    return final_window


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    trans_coor = TRANSFORM_COOR_LAYERS * pos
    return np.apply_along_axis(sample_descriptor_one_point, COL, trans_coor, im,
                               desc_rad)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
          1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
          2) A feature descriptor array with shape (N,K,K)
    """
    points = spread_out_corners(pyr[PYR_FIRST_LEVEL], 7, 7, 20)
    desc = sample_descriptor(pyr[PYR_3RD_LEVEL], points, 3)
    return [points, desc]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # create the S matrix
    S = desc1.reshape(len(desc1), -1).dot(desc2.reshape(len(desc2), -1).T)

    # create true&false matrices according to the terms
    S_min = (S > min_score)
    S_max_row = (S >= (np.sort(S)[:, SECOND_MAX].reshape((len(S), 1))))
    S_max_col = (S >= (np.sort(S, axis=0)[SECOND_MAX, :].reshape(1, len(S[0]))))

    match_matrix = S_min * S_max_row * S_max_col
    return [np.where(match_matrix)[0], np.where(match_matrix)[1]]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates
    obtained from transforming pos1 using H12.
    """
    add_coor = np.ones((pos1.shape[0], 1))
    three_coor = np.hstack((pos1, add_coor)).T
    new_coor = (H12 @ three_coor).T
    return new_coor[:, [0, 1]] / new_coor[:, [2, 2]]


def calculate_Ej(H, points1, points2):
    """
    This function calculates the euclidean distance for a given 2 points array
    :param H: A 3x3 homography matrix.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates
    of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates
    of matched points in image 2.
    :return: the euclidean distance for points2 and points2'(after applying homography
    """
    p2_homo = apply_homography(points1, H)
    p2_homo_minos_p2 = p2_homo - points2
    Ej = np.power(np.linalg.norm(p2_homo_minos_p2, axis=1), POWER)
    return Ej


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates
    of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates
    of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier
                  matches found.
    """
    random_pairs = []
    max_inliers = []
    for i in range(num_iter):
        rand_j = random.randrange(0, len(points1))
        random_pairs.append([np.array(points1[rand_j]), np.array(points2[rand_j])])

        # if the case is not only translation, add another random point:
        if not translation_only:
            rand_j_2 = random.randrange(0, len(points1))
            while rand_j == rand_j_2:
                rand_j_2 = random.randrange(0, len(points1))
            random_pairs.append([points1[rand_j_2], points2[rand_j_2]])

        # create arrays for the random points
        points1_j = np.array(random_pairs)[:, 0, :]
        points2_j = np.array(random_pairs)[:, 1, :]

        # find the inliers set and homography
        j_inliers, H = ransac_method(inlier_tol, points1, points1_j, points2,
                                     points2_j, translation_only)

        # update the inliers list
        if len(max_inliers) == 0:
            max_inliers = j_inliers
        elif len(np.where(j_inliers)[0]) > len(np.where(max_inliers)[0]):
            max_inliers = j_inliers
        random_pairs = []

    # calculate the final inliers set and homography after the iterations
    j_in = np.where(max_inliers)[0]
    p1Jin = np.take(points1, j_in, axis=0)
    p2Jin = np.take(points2, j_in, axis=0)

    j_inliers, H = ransac_method(inlier_tol, points1, p1Jin, points2, p2Jin,
                                 translation_only)
    return [H, np.where(j_inliers)[0].reshape((len(np.where(j_inliers)[0]),))]


def ransac_method(inlier_tol, points1, points1_j, points2, points2_j,
                  translation_only):
    """
    This function operates the 3 steps of ransac
    :param inlier_tol: inlier tolerance threshold.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates
        of matched points in image 1.
    :param points1_j: an array with random points from points1
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates
        of matched points in image 2.
    :param points2_j: an array with random points from points2
    :param translation_only: see estimate rigid transform
    :return: j_inliers- a boolean array- true indices mean inlier point from the
                    given points array
            H - A 3x3 array with the computed homography.
    """
    H = estimate_rigid_transform(points1_j, points2_j, translation_only)
    Ej = calculate_Ej(H, points1, points2)
    j_inliers = np.array(Ej < inlier_tol)
    return j_inliers, H


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of
    matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of
    matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    combine_im = np.hstack((im1, im2))

    # fit image2 coordinates
    moved_points2 = points2
    moved_points2[:, 0] += im1.shape[1]

    outliers_points1 = np.delete(points1, inliers, axis=0)
    outliers_points2 = np.delete(moved_points2, inliers, axis=0)

    plt.figure()
    plt.imshow(combine_im, cmap='gray')

    # plot the outliers
    plt.plot((outliers_points1[:, 0], outliers_points2[:, 0]),
             (outliers_points1[:, 1], outliers_points2[:, 1]), mfc='r', c='b', lw=.2,
             ms=1, marker='o')

    # plot the inliers
    plt.plot((points1[inliers][:, 0], moved_points2[inliers][:, 0]),
             (points1[inliers][:, 1], moved_points2[inliers][:, 1]), mfc='r',
             c='y', lw=.4, ms=1, marker='o')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of successive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
        matrices where H_successive[i] is a homography which transforms points
        from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
        accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = [1] * (len(H_succesive) + 1)
    H2m[m] = np.eye(IDENTITY_MATRIX)

    # for the image indices smaller then m
    for beg in range(m - 1, -1, -1):
        H2m[beg] = H2m[beg + 1] @ H_succesive[beg]

    # for the image indices larger then m
    for end in range(m + 1, len(H2m)):
        H2m[end] = H2m[end - 1] @ np.linalg.inv(H_succesive[end - 1])

    # normalize the result
    for i in range(len(H2m)):
        H2m[i] = H2m[i] / H2m[i][2, 2]
    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping
    the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    image_edges = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    trans_image_edges = np.array(apply_homography(image_edges, homography))
    min_x = np.min(trans_image_edges[:, 0])
    max_x = np.max(trans_image_edges[:, 0])
    min_y = np.min(trans_image_edges[:, 1])
    max_y = np.max(trans_image_edges[:, 1])
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    w = image.shape[1]
    h = image.shape[0]

    top_left_corner, bottom_right_corner = compute_bounding_box(homography, w, h)

    x_cord_after = np.arange(top_left_corner[0], bottom_right_corner[0])
    y_cord_after = np.arange(top_left_corner[1], bottom_right_corner[1])

    Xcord, Ycord = np.meshgrid(x_cord_after, y_cord_after)

    points = np.column_stack((Xcord.flatten(), Ycord.flatten()))
    trans_points = apply_homography(points, np.linalg.inv(homography))
    warped_image = map_coordinates(image, [trans_points[:, 1].reshape(Ycord.shape),
                                           trans_points[:, 0].reshape(Xcord.shape)],
                                   order=1, prefilter=False)
    return warped_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for
                      i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system
        # of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i],
                                                          self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input
        # images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the
            # middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x coordinate of each slice
            # center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - \
                                      global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = (
                (warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:,
                              boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom,
                boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most
        # image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
