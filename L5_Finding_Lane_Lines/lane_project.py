#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
import os
image_list = os.listdir("test_images/")
image_list[0]

## Read in an Image

#reading in an image
#for i in range(0, len(image_list)):
    
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

## Ideas for Lane Detection Pipeline
'''
**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**
'''

## Helper Functions
#Below are some helper functions to help get you started. They should look familiar from the lesson!

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
    `vertices` should be a numpy array of integer points.
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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
    
    left_line = []
    right_line= []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            # use polyfit function to find out the slope and y_intercept points to connect the dotted lines.
            poly_param = np.polyfit((x1,x2), (y1,y2), 1)
            m_slope, y_intercept = poly_param[0], poly_param[1]
            
            
            if m_slope < 0: # As y increase downwards on cv2 image coordinates, the slope on the left side is negative
                left_line.append((m_slope, y_intercept))              
            else: # As x and y decrease at the same time, the slope on the right side is positive
                right_line.append((m_slope, y_intercept))
    
    # get the average slope and intercept from line list to make one line on each side.
    left_line_average = np.average(left_line, axis = 0)
    right_line_average = np.average(right_line, axis = 0)
    
    # final slope and intercept value of the lines
    left_slope, left_intercept = left_line_average
    right_slope, right_intercept = right_line_average
    
    # get the end points of the each line within the region of interest
    left_line_x_bottom = (y_max - left_intercept)/left_slope
    left_line_y_bottom = left_line_x_bottom * left_slope + left_intercept
                        
    left_line_x_top = (y_min - left_intercept)/left_slope
    left_line_y_top = left_line_x_top * left_slope + left_intercept
    
    right_line_x_bottom = (y_max - right_intercept)/right_slope
    right_line_y_bottom = right_line_x_bottom * right_slope + right_intercept
        
    right_line_x_top = (y_min - right_intercept)/right_slope
    right_line_y_top = right_line_x_top * right_slope + right_intercept
    
    # made two colours to represent the each line separately
    green = [0, 255, 0]
    red = [255, 0, 0]
    
    # draw the left line
    cv2.line(img, (left_line_x_bottom.astype(np.int32),left_line_y_bottom.astype(np.int32)),
             (left_line_x_top.astype(np.int32),left_line_y_top.astype(np.int32)), green, thickness)
    # draw the right line
    cv2.line(img, (right_line_x_bottom.astype(np.int32),right_line_y_bottom.astype(np.int32)),
               (right_line_x_top.astype(np.int32),right_line_y_top.astype(np.int32)), red, thickness)

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

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

## Test Images

#Build your pipeline to work on the images in the directory "test_images"  
#**You should make sure your pipeline works well on these images before you try the videos.**

import os
image_list = os.listdir("test_images/")
image_list[0]

## Build a Lane Finding Pipeline
# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory,
# and you can use the images in your writeup report.
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# convert the image to grayscale
gray = grayscale(image)
plt.imshow(gray, cmap = 'gray')

# Define a kernal_size for gaussian smoonthing / blurring
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)


# Define threshold parameter for canny edge detecion 
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)

# create a masked edges image using cv2.fillpoly()
#mask = np.zeros_like(edges)
#ignore_mask_color = 255


# Define a four sided polygon to mask
imshape = image.shape
y_max = imshape[0]
x_max = imshape[1]
y_min = 310

vertices = np.array([[(0,y_max),(480, y_min), (490, y_min), (x_max,y_max)]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)


# Display the image
#plt.imshow(edges, cmap = 'Greys_r')


# Hough 
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 5
max_line_gap = 3
line_image = np.copy(image) * 0

lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
line_edges = weighted_img(lines, image)
plt.imshow(line_edges)

## Test on Videos

'''
You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). 
Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**
'''

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Define a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(480, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    
    # Display the image
    #plt.imshow(edges, cmap = 'Greys_r')
    
    
    # Hough 
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 5
    max_line_gap = 3
    line_image = np.copy(image) * 0
    
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    result = weighted_img(lines, image)
    
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
clip1

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

