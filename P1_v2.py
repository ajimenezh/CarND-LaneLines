
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML

class Line:
	
	def __init__(self, slope, point):
		self.slope = slope
		self.point = point

	def get_y(self, x):
		if abs(self.slope) < 1.0e-6:
			return self.point[1]
		return (x - self.point[0])/self.slope + self.point[1]

	def get_x(self, y):
		return (y - self.point[1])*self.slope + self.point[0]

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

def average_line(lines):
	
	avg_m = 0.0

	for line in lines:
		avg_m += line.slope

	avg_y0 = 0.0
	for line in lines:
		avg_y0 += line.get_y(0)
	
	n = max(len(lines), 1)

	return Line(avg_m/n, [0, avg_y0/n])

def median_line(lines):
	
	l = []

	for line in lines:
		l.append(line.slope)

	y = []
	for line in lines:
		y.append(line.get_y(0))
	
	n = len(lines)

	l.sort()
	y.sort()

	if n == 0:
		return Line(0, [0,0])
	else:
		return Line(l[int(n/2)], [0, y[int(n/2)]])

globalError = False
globalLines = [[Line(0.0, [0.0,0.0]), Line(0.0, [0.0,0.0])]]

def draw_lines(img, lines, color=[255, 0, 0, 0], thickness=12):
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

	imshape = img.shape

	global globalLines

	test = False
	if test:
		pi = math.acos(-1.0)
		low = 20
		hi = 70
		for line in lines:
			for x1,y1,x2,y2 in line:
				ang = math.atan2(x2-x1, y2-y1)
				#print(x2-x1, y2-y1, ang)
				if (ang>low/180.0*pi and ang<hi/180.0*pi) or (ang>(90+low)/180.0*pi and ang<(hi+90)/180.0*pi):
					m = ((float(x2)-x1)/(float(y2)-y1))
					l = Line(m, [x1, y1])
					
					if (m < 0.0 and l.get_x(imshape[0]) < 4*imshape[1]/10) or \
 						(m > 0.0 and l.get_x(imshape[0]) > 6*imshape[1]/10):
							cv2.line(img, (x1, y1), (x2, y2), color, thickness)
	else:
		linesData = []

		global globalError
		globalError = False

		pi = math.acos(-1.0)
		low = 20
		hi = 70

		for line in lines:
			tmp = []
			for x1,y1,x2,y2 in line:
				if y1 != y2:
					ang = math.atan2(x2-x1, y2-y1)
					if (ang>low/180.0*pi and ang<hi/180.0*pi) or (ang>(90+low)/180.0*pi and ang<(hi+90)/180.0*pi):
						m = ((float(x2)-x1)/(float(y2)-y1))
						l = Line(m, [x1, y1])
					
						if (m < 0.0 and l.get_x(imshape[0]) < 4*imshape[1]/10) or \
	 						(m > 0.0 and l.get_x(imshape[0]) > 6*imshape[1]/10):
								linesData.append(l)
	
		imshape = img.shape

		lines_right = []
		lines_left = []
		for line in linesData:
			if line.get_x(imshape[0]) < imshape[1]/2:
				lines_left.append(line)
			else:
				lines_right.append(line)
	
		line_right = median_line(lines_right)
		line_left = median_line(lines_left)

		globalLines.append([line_right, line_left])

		if abs(line_right.slope) < 1.0e-6:
			print ("try")
			if abs(globalLines[-2][0].slope) > 1.0e-6:
				print (globalLines[-2][0].slope)
				line_right = globalLines[-2][0]
			elif len(globalLines) > 2 and abs(globalLines[-3][0].slope) > 1.0e-6:
				print (globalLines[-3][0].slope)
				line_right = globalLines[-3][0]

		if line_left.slope == 0 and globalLines[-2][1].slope != 0:
			line_left = globalLines[-2][1]

		cv2.line(img, (int(line_right.get_x(imshape[0])), imshape[0]), (int(line_right.get_x(12*imshape[0]/20)), int(12*imshape[0]/20)), color, thickness)

		cv2.line(img, (int(line_left.get_x(imshape[0])), imshape[0]), (int(line_left.get_x(12*imshape[0]/20)), int(12*imshape[0]/20)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

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

def show(img):
	plt.imshow(img, cmap='gray')
	plt.show()

idx = 0

def process_image(image):
	gray = image

	kernel_size = 7
	blur_gray = gaussian_blur(gray,kernel_size)

	#show(blur_gray)
	
	low_threshold = 80 # low between 1:2 and 1:3 of high 
	high_threshold = 160
	edges = canny(blur_gray, low_threshold, high_threshold)

	imshape = image.shape

	vertices = np.array([[(0,imshape[0]),(24*imshape[1]/50, 12*imshape[0]/20), (26*imshape[1]/50, 12*imshape[0]/20), (imshape[1],imshape[0])]], dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)
	
	#show(masked_edges)
	
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 5     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 30 #minimum number of pixels making up a line
	max_line_gap = 20    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on
	
	line_img, lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
	
	#show(line_img)


	global idx
	cv2.imwrite("test_videos_output/test_" + str(idx) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	idx += 1

	result_image = weighted_img(line_img, image)

	cv2.imwrite("test_videos_output/res_" + str(idx) + ".jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

	return result_image

TEST = True
if TEST:
	path = "test_images/"
	path = "test/"
	imglist = os.listdir(path)

	print (imglist)

	for file in imglist[:]:
	
		image = 	mpimg.imread(path + file)
		result_image = process_image(image)

		show(result_image)

else:
	white_output = 'test_videos_output/challenge.mp4'
	## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
	## To do so add .subclip(start_second,end_second) to the end of the line below
	## Where start_second and end_second are integer values representing the start and end of the subclip
	## You may also uncomment the following line for a subclip of the first 5 seconds
	##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
	clip1 = VideoFileClip("test_videos/challenge.mp4")
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)
	
