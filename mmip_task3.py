# -*- coding: utf-8 -*-

from skimage import io
import numpy as np
from numpy import linalg as LA
from sys import argv

def myconvolve(img, kernel):
    # calc the size of the array of submatracies
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatracies
    submatrices = strd(img,kernel.shape + sub_shape,img.strides * 2)

    # sum the submatraces and kernel
    convolved_matrix = np.einsum('ij,ijkl->kl', kernel, submatrices)

    return convolved_matrix


def Gauss_gradient(sigma):
    # sigma --> int
    s = sigma
    sigma = int(np.around(sigma))
    
    # fill the matrix with algebraic calculated derivatives of 2D-Gauss function
    Gauss_x = np.array([np.array([ -1 * x * np.exp(-(x*x + y*y)/ (2 * s*s))
                for y in range(-3*sigma, 3*sigma+1)]) for x in range(-3*sigma, 3*sigma+1)])

    Gauss_y = np.array([np.array([ - 1  * y * np.exp(-(x*x + y*y)/ (2 *s*s))
                for y in range(-3*sigma, 3*sigma+1)]) for x in range(-3*sigma, 3*sigma+1)])

    return Gauss_x, Gauss_y


def gradient_magnitude(image, sigma):
    # get Gauss derivatives
    Gauss_x, Gauss_y = Gauss_gradient(sigma)

    # sigma --> int
    sigma = int(np.around(sigma))
    
    # extend image in order to work with borders
    big_image = np.pad(image, pad_width=3*sigma, mode='edge')

    # convolve extemded image with Gauss derivatives
    img_x = myconvolve(big_image, Gauss_x)
    img_y = myconvolve(big_image, Gauss_y)

    # calc gradient magnitude 
    gradient_magnitude = np.hypot(img_x, img_y)

    # normalize gradient magnitude to [0, 255]
    grad_max = np.max(gradient_magnitude)
    gradient_magnitude *= 255 / grad_max

    # return grad magn and convolved images
    return gradient_magnitude, img_x, img_y


def non_max_suppression(image, sigma):
    # get gradient magnitude and convolved images 
    GM, img_x, img_y = gradient_magnitude(image, sigma)

    # sigma --> int
    sigma = int(np.around(sigma))
    
    # calc gradient direction 
    GD = np.arctan2(img_y, img_x)

    # rescale gradient directon from [-pi, pi] to [0, 360]
    GD = GD * 180 / np.pi + 180

    # extend image in order to work with borders
    big_image = np.pad(GM, pad_width=1, mode='edge')
    
    # prepare base for the result image
    nms = GM.copy()

    # for every pixel of the image calculate: ...
    for x in range(1, GM.shape[0] + 1):
        for y in range(1, GM.shape[1] + 1):
            # current pixel value
            c_pixel = big_image[x, y]
            # current value of gradient direction
            c_dir = GD[x-1, y-1]
            # addition coordinates to get neighbours in the gradient direction
            add_x = None
            add_y = None
            # some manipulations with numbers
            if (c_dir >= 337.5) | (c_dir <= 22.5) | (157.5 <= c_dir < 202.5):
                add_x, add_y = (1, 0)
            elif (22.5 <= c_dir < 67.5) | (202.5 <= c_dir < 247.5):
                add_x, add_y = (1, 1)
            elif (67.5 <= c_dir < 112.5) | (247.5 <= c_dir < 292.5):
                add_x, add_y = (0, 1)
            else:
                add_x, add_y = (1, -1)
            # if current pixel IS NOT a local maximum => suppress it
            if (c_pixel <= big_image[x+add_x, y+add_y]) | (c_pixel < big_image[x-add_x, y-add_y]):
                nms[x-1, y-1] = 0

    # return image with suppressed pixels
    return nms


def canny_edge_detector(image, sigma, thresholds):
    # get image with edges from NMS
    nms_image = non_max_suppression(image, sigma)

    # sigma --> int
    sigma = int(np.around(sigma))
    
    # set params
    high = float(thresholds[0]) * 255
    low = float(thresholds[1]) * 255

    ### first step of threshold ###

    # where pixels are brighter than high => set maximum (255)
    nms_image[nms_image > high] = 255
    # where pixels are dimmer than low => set minimum (0)
    nms_image[nms_image < low] = 0
    # else => set meanvalue (127)
    nms_image[(nms_image <= high) & (nms_image >= low)] = 127

    ### second step of threshold ###

    # extend the image in order to work with borders
    big_image = np.pad(nms_image, pad_width=2, mode='edge')
    
    # prepare base for the result image
    canny_im = nms_image.copy()

    # for every pixel in image calculate: ... 
    for x in range(2, canny_im.shape[0] + 2):
        for y in range(2, canny_im.shape[1]+ 2):
            # find 'middle' pixels
            if big_image[x,y] == 127:
                # take 3x3 area around found pixel
                curr_fold = big_image[x-1:x+2, y-1:y+2]
                # if any strong pixels (255) in the area => our pixel is strong (set 255)
                # else => our pixel is weak => suppress it (set 0)
                canny_im[x-2,y-2] = 255 if np.any(curr_fold == 255) else 0

    # return result image for canny algorithm 
    return canny_im

def Gauss_sec_der(sigma):
    # get first Gauss derivatives
    Gauss_x, Gauss_y = Gauss_gradient(sigma)

    # get second Gauss derivatives 
    Gauss_x_x, Gauss_x_y = np.gradient(Gauss_x)
    Gauss_x_y, Gauss_y_y = np.gradient(Gauss_y)

    # return second Gauss derivatives
    return Gauss_x_x, Gauss_x_y, Gauss_y_y

def get_eig(image, sigma):
    # s = sigma --> int
    s = int(np.around(sigma))

    # extend image in order to work with borders
    big_image = np.pad(image, pad_width=3*s, mode='edge')

    # get Gauss second derivatives 
    Gauss_x_x, Gauss_x_y, Gauss_y_y = Gauss_sec_der(sigma)

    # convolve image with Gauss second derivatives
    I_xx = myconvolve(big_image, Gauss_x_x)
    I_xy = myconvolve(big_image, Gauss_x_y) 
    I_yy = myconvolve(big_image, Gauss_y_y)

    # empty matrixes for hessian eigen values(w) and eigen vectors(v)
    hessian_eig_w = np.empty(image.shape)
    hessian_eig_v = np.empty(image.shape, dtype='object')

    # for every pixel in image calculate: ... 
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):   
            # hessian = [[I_xx, I_xy], [I_xy, I_yy]]
            c_hessian = np.array([[I_xx[x,y], I_xy[x,y]], [I_xy[x,y], I_yy[x,y]]])
            # calculate current eig_w = [w1, w2] and eig_v = [v1, v2]
            c_eig_w, c_eig_v = LA.eig(c_hessian)
            # get index of max abs eig_w
            eig_w_argmax = np.argmax(np.abs(c_eig_w))
            # vessels are darker => suppress eig_w: eig_w < 0
            hessian_eig_w[x, y] = c_eig_w[eig_w_argmax] if c_eig_w[eig_w_argmax] > 0 else 0
            # get rounded eig_vector corresponding to max abs eig_w
            hessian_eig_v[x, y] = np.around(c_eig_v[:, eig_w_argmax]).astype('int')
    
    # return hessian eigen values(w) and eigen vectors(v) for the image
    return hessian_eig_w, hessian_eig_v


def eig_nonmax_suppression(image, sigma):
    # get eigen values(w) and eigen vectors(v)
    eig_w, eig_v = get_eig(image, sigma)

    # extend eig_w matrix in order to work with borders
    big_eig_w = np.pad(eig_w, pad_width=1, mode='edge')

    # for every pixel in image calculate: ... 
    for x in range(1, eig_w.shape[0]+1):
        for y in range(1, eig_w.shape[1]+1):
            # current pixel value = max abs eigen value in (x, y)
            c_pixel = eig_w[x-1, y-1]
            # current direction = eigen vector for max abs eigen value
            c_dir = eig_v[x-1, y-1]
            # pixels in current direction
            a_pixel = big_eig_w[x + c_dir[0], y + c_dir[1]]
            b_pixel = big_eig_w[x - c_dir[0], y - c_dir[1]]
            # suppress pixel if it's not a local maximum
            if c_pixel <= max(a_pixel, b_pixel):
                eig_w[x-1, y-1] = 0
    
    # normalize result to [0, 255]
    eig_w_max = np.amax(eig_w)
    eig_w *= 255 / eig_w_max

    # return normalized image with nonmax suppressed
    return eig_w
    


def vessels(image):
    # in ridges result image will be returned 
    ridges = np.empty(image.shape)

    # get 3 ridge images with different sigma
    nonmax2 = eig_nonmax_suppression(image, 2.2)
    nonmax3 = eig_nonmax_suppression(image, 3.1)
    nonmax4 = eig_nonmax_suppression(image, 4.0)

    # result image = unity of images with diff. sigma
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            ridges[x,y] = max(nonmax2[x,y],nonmax3[x,y], nonmax4[x,y])

    # return result image
    return ridges


command = argv[1]
params = tuple(argv[2:len(argv)-2])
input_file = argv[len(argv)-2]
output_file = argv[len(argv)-1]

image = io.imread(input_file, as_gray=True)

if command == 'vessels':
    result_image = vessels(image)
else:
    sigma = float(params[0])

    if command == 'grad':
        result_image = gradient_magnitude(image, sigma)[0]
    elif command == 'nonmax':
        result_image = non_max_suppression(image, sigma)
    else:
        result_image = canny_edge_detector(image, sigma, params[1:])

io.imsave(output_file, result_image.astype('uint8'))
