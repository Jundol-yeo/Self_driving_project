import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import matplotlib.patches as patches

%matplotlib inline

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images.

images = glob.glob('./camera_cal/calibration*.jpg')
img_detected_corners = []

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        img_detected_corners.append(img)

        #plt.show()
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()
plt.figure(figsize=(50,30))
columns = 4
for i, image in enumerate(img_detected_corners):
    plt.subplot(len(img_detected_corners) / columns + 1, columns, i + 1)
    plt.imshow(image)



def cal_undistort(img, objpoints, imgpoints):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh_min = 0, thresh_max = 255):
    # Convert the image to RGB or BGR to Gray sacle image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # If image was read using cv2.imread, COLOR_BGR2GRAY should be used
    
    # take the derivative in x or y from given orient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Calculate the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # Scale to 8 bit (0 - 255) then convert to type of np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    # Create a mask of 1's where the scaled gradient magnitude
    binary_output = np.zeros_like(scaled_sobel)
    
    # return the mask as output
    binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel = 3, mag_thresh = (0, 255)):
    
    # convert the image to the gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # take the gradient of x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Calculate the magnitude
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale to 8-bit (0 - 255) and convert to type of uint6
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    
    # Return the mask as the output image
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    
    print(binary_output.shape)
    
    return binary_output

def dir_threshold(img, sobel_kernel = 3, thresh = (0, np.pi / 2)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    binary_output = np.zeros_like(grad_dir)
    
    binary_output[(grad_dir > thresh[0]) & (grad_dir < thresh[1])] = 1
    
    return binary_output

def hls_select(img, thresh = (0, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    
    h_binary = np.zeros_like(h)
    h_binary[(h > thresh[0]) & (h <= thresh[1])] = 1
    
    l_binary = np.zeros_like(l)
    l_binary[(l > thresh[0]) & (l <= thresh[1])] = 1
    
    s_binary = np.zeros_like(s)
    s_binary[(s > thresh[0]) & (s <= thresh[1])] = 1

    return h_binary, l_binary, s_binary

def hls_select_specific(img, channel, thresh = (0, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)#
    channel = hls[:, : ,channel]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def corners_unwarp(img):
    
    src = np.float32([[230, 700], [600, 440],
                    [680, 440], [1100, 700]])

    dst = np.float32([[250, 720], [250, 0],
                    [1100, 0], [1100, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    
    return warped, M, M_inv 

def combined_threshold(undist_img):

    kernel_size = 7
    gradx = abs_sobel_thresh(undist_img, orient = 'x', sobel_kernel = kernel_size, thresh_min = 50, thresh_max = 255)
    grady = abs_sobel_thresh(undist_img, orient = 'y', sobel_kernel = kernel_size, thresh_min = 50, thresh_max = 255)
    mag_thresh_output = mag_thresh(undist_img, sobel_kernel = 15, mag_thresh = (25, 255))
    dir_thresh_output = dir_threshold(undist_img, sobel_kernel = 7, thresh = (0.5, 0.9))
    s_output = hls_select_specific(undist_img, channel = 2, thresh = (125, 255))
    
    #print(gradx)
    #print(grady)
    #print(mag_thresh_output.shape)
    #print(dir_thresh_output.shape)
    #print(undist_img.shape)
    
    combined_result = np.zeros_like(dir_thresh_output)
    combined_result[((gradx == 1) & (grady == 1)) | ((mag_thresh_output == 1) & (dir_thresh_output == 1)) | s_output == 1] = 1
    #combined_result[ s_output == 1 | ((mag_thresh_output == 1) & (dir_thresh_output == 1)) ] = 1
    
    return combined_result

def hist(img):
    
    # Grab the bottom half of the image (Y value increases)
    
    bottom_half = img[img.shape[0]//2:, :]
    
    histogram = np.sum(bottom_half, axis = 0)
    
    # print(bottom_half.shape)
    # print(histogram.shape)
    # print(histogram[:340])
    
    return histogram


def find_lane_pixels(img):
    
    histogram = hist(img)
    
    # Create an output image to draw on and visualise the result
    out_img = np.dstack((img, img, img))
    print(out_img.shape)
    
    plt.imshow(out_img)
    plt.show()
    
    # Find the peak of the left side and right side of the histogram using midpoint
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Hyper Parameters
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the window +/- margin
    margin = 150
    # Set minimum number of pixels found to recentre window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to recieve elft and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and left and right)
        win_y_low = img.shape[0] - (window+1) * window_height
        win_y_high = win_y_low + window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        #plt.imshow(out_img)
        #plt.show()
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(img):
    
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
         # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    print(leftx)
    
    #plt.imshow(out_img)
    #plt.show()

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    print(left_fit[0])
    print(right_fit)
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(img):
    
        # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(img)
    
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (
        (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
        (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))
        )
    right_lane_inds = (
        (nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
        (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))
        )
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result


def fill_inside_poly(warped_gray_img, original_img):
    
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(warped_gray_img)
    
    warped_zero = np.zeros_like(warped_gray_img)
    color_warp = np.dstack((warped_gray_img, warped_gray_img, warped_gray_img))
    
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    overall_pts = np.hstack((left_line_pts, right_line_pts))
    
    cv2.fillPoly(color_warp, np.int_([overall_pts]), (0, 255, 0))
    
    final_warp = cv2.warpPerspective(color_warp, M_inv, (original_img.shape[1], original_img.shape[0]))
    
    result = cv2.addWeighted(original_img, 1, final_warp, 0.3, 0)
    
    return result

def radius_and_offset(left_fit, right_fit, warped_img):

    # Convert x and y pixel space to meter space according based on images and road conditions
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    
    # Generate y data to represnet lane-line pixel based on the image shape
    y_points = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    
    # Generate polynomial equation on the left curve and the right curve separately
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_fitx = left_fitx[::-1]
    right_fixt = right_fitx[::-1]
    
    left_fit_cr = np.polyfit(y_points*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_points*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y value where we want radius of curvature
    y_eval = np.max(ploty)
    
    ## Implement the calculation of the left and right line here
    left_curverad = ((1 + (2*left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]* y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    average_curverad = (left_curverad + right_curverad) / 2
    
    ### Centre line
    bottom_y = image.shape[0]
    
    # Generate polynomial
    left_fit_bottom = left_fit[0]*bottom_y**2 + left_fit[1]*bottom_y + left_fit[2]
    right_fit_bottom = right_fit[0]*bottom_y**2 + right_fit[1]*bottom_y + right_fit[2]
    
    lane_centre = (left_fit_bottom + right_fit_bottom)/2
    
    offset_pix = image.shape[1]/2 - lane_centre
    offset_m = offset_pix*xm_per_pix
    
    return left_curverad, right_curverad, average_curverad, offset_m