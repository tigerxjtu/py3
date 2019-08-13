#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
os.listdir("test_images/")


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None:
        lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


#img_names = ['solidWhiteCurve', 'solidWhiteRight', 'solidYellowCurve',
#             'solidYellowCurve2', 'solidYellowLeft', 'whiteCarLaneSwitch',
#             'challenge_shadows', 'challenge_light_cement']

img_names = ['solidWhiteCurve']

def median(items, value=(lambda x: x), weight=(lambda x: 1.0)):
    pairs = [[value(x), weight(x)] for x in sorted(items, key=value)]
    total_weight = sum([x[1] for x in pairs])
    half_weight = total_weight * 0.5
    weight_so_far = 0.0
    for pair in pairs:
        weight_so_far = weight_so_far + pair[1]
        if weight_so_far > half_weight:
            return pair[0]
    return None


def unnest_lines(lines):
    result = []
    for [[x1, y1, x2, y2]] in lines:
        result.append([x1, y1, x2, y2])
    return result


def nest_lines(lines):
    result = []
    for [x1, y1, x2, y2] in lines:
        result.append([[x1, y1, x2, y2]])
    return result


def filter_verticalish(lines):
    result = []
    for line in lines:
        x1, y1, x2, y2 = line
        if abs(x1 - x2) < 2 * abs(y1 - y2):
            result.append(line)
    return result


def line_center(line):
    return [0.5 * (line[0] + line[2]), 0.5 * (line[1] + line[3])]


def line_length(line):
    return ((line[0] - line[2]) ** 2.0 + (line[1] - line[3]) ** 2.0) ** 0.5


def filter_left_right(lines, xcenter):
    left = []
    right = []
    for line in lines:
        if line_center(line)[0] < xcenter:
            left.append(line)
        else:
            right.append(line)
    return left, right


def inverse_slope(line):
    deltax = line[0] - line[2]
    deltay = line[1] - line[3]
    return deltax * 1.0 / deltay


def x_intercept(point, inverse_slope):
    return point[0] - inverse_slope * point[1]


def median_inverse_slope(lines):
    return median(lines, value=inverse_slope, weight=line_length)


def median_line(lines, ymin, ymax):
    minv = median_inverse_slope(lines)
    xo = median(lines, value=(lambda x: x_intercept(x, minv)), weight=line_length)
    return [int(xo + ymin * minv), ymin, int(xo + ymax * minv), ymax]


def edge_lines_to_lane_lines(lines, xmin, ymin, xmax, ymax):
    lines = unnest_lines(lines)
    lines = filter_verticalish(lines)
    left, right = filter_left_right(lines, 0.5 * (xmin + xmax))
    lines = []
    if len(left) > 0:
        lines.append(median_line(left, ymin, ymax))
    if len(right) > 0:
        lines.append(median_line(right, ymin, ymax))
    return nest_lines(lines)


def xform_img(img):
    original = img

    b, g, r = cv2.split(img)

    img = grayscale(img)
    img = gaussian_blur(img, 5)
    img = canny(img, 50, 150)

    plt.imshow(img)
    plt.show()

    for channel in [b, g, r]:
        channel = gaussian_blur(channel, 5)
        channel = canny(channel, 50, 150)
        img = weighted_img(img, channel, 1., 1.)

    top = 0
    bottom = img.shape[0]
    left = 0
    right = img.shape[1]
    horizon = int(0.4 * top + 0.6 * bottom)
    width = right - left
    xcenter = (left + right) / 2

    vertices = [(left, bottom),
                (xcenter - width / 30, horizon),
                (xcenter + width / 30, horizon),
                (right, bottom)]

    img = region_of_interest(img, np.int32(np.array([vertices])))

    plt.imshow(img)
    plt.show()

    rho = 2
    theta = 1 * np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 40

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    img_copy = img
    draw_lines(img_copy,lines)
    plt.imshow(img_copy)
    plt.show()


    lanes = edge_lines_to_lane_lines(lines, xmin=left, ymin=horizon, xmax=right, ymax=bottom)
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(img, lanes)

    img = weighted_img(img, original)

    return img


for img_name in img_names:
    img = mpimg.imread('test_images/' + img_name + '.jpg')
    transformed = xform_img(img)
    mpimg.imsave('test_images/' + img_name + '_out.jpg', transformed, cmap='gray')
    plt.imshow(transformed)
    plt.show()