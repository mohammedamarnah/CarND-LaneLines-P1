import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
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
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
    top = 320
    bottom = 550

    left = [[0]*0 for i in range(4)]
    right = [[0]*0 for i in range(4)]
    
    # left1[0].append(10)
    # left1[1].append(20)
    # print(left1)

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue        
                left[0].append(x1)
                left[1].append(y1)
                left[2].append(x2)
                left[3].append(y2)
            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue        
                right[0].append(x1)
                right[1].append(y1)
                right[2].append(x2)
                right[3].append(y2)
    
    # LEFT LINE
    avgL = []
    avgL.append(int(np.mean(left[0])))
    avgL.append(int(np.mean(left[1])))
    avgL.append(int(np.mean(left[2])))
    avgL.append(int(np.mean(left[3])))
    left_slope = get_slope(avgL[0], avgL[1], avgL[2], avgL[3])

    leftline = []
    # X1 
    leftline.append(int(avgL[0] + (top - avgL[1]) / left_slope))
    # Y1
    leftline.append(top)
    # X2
    leftline.append(int(avgL[0] + (bottom - avgL[1]) / left_slope))
    # Y2
    leftline.append(bottom)
    cv2.line(img, (leftline[0], leftline[1]), (leftline[2], leftline[3]), color, thickness)   


    # RIGHT LINE
    avgR = []
    avgR.append(int(np.mean(right[0])))
    avgR.append(int(np.mean(right[1])))
    avgR.append(int(np.mean(right[2])))
    avgR.append(int(np.mean(right[3])))
    right_slope = get_slope(avgR[0], avgR[1], avgR[2], avgR[3])

    rightline = []
    # X1
    rightline.append(int(avgR[0] + (top - avgR[1]) / right_slope))
    # Y1
    rightline.append(top)
    # X2
    rightline.append(int(avgR[0] + (bottom - avgR[1]) / right_slope))
    # Y2
    rightline.append(bottom)
    cv2.line(img, (rightline[0], rightline[1]), (rightline[2], rightline[3]), color, thickness)     
    
def get_slope(x1, y1, x2, y2):
    return ((y2-y1)/(x2-x1))

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
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

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

'''
TODO_ME: 
    Edit the MASKED REGION, 
    and then the MAX LINE LENGTH, 
    and the MAX GAP LENGTH.
'''

import os

def process_image(image):
    # you should return the final output (image with lines are drawn on lanes)
    gray = grayscale(image)
    gaus = gaussian_blur(gray, 5)
    edges = canny(gaus, 50,150)    
    imshape = image.shape
    
    vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)    
    masked = region_of_interest(edges, vertices)
    
    rho = 2            #distance resolution in pixels of the Hough grid
    theta = np.pi/180  #angular resolution in radians of the Hough grid
    threshold = 20     #minimum number of votes (intersections in Hough grid cell)
    min_line_len = 25  #minimum number of pixels making up a line
    max_line_gap = 10  #maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
    
    result = weighted_img(line_image, image)
    return result

for file in os.listdir("test_images/"):
    img = mpimg.imread("test_images/"+file)
    img_lines = process_image(img)
    # Display the image
    plt.imshow(img_lines)
    cv2.imwrite("test_images_output/"+file, img_lines)