import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	bounding_box = ((0,0),(0,0))

	# optimized implementation
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_orange_bound = np.array([0, 200, 102]) #BEST 0.88221
	upper_orange_bound = np.array([30, 255, 255])
	mask = cv2.inRange(hsv_img, lower_orange_bound, upper_orange_bound)
		
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
		bounding_box = ((x, y), (x + w, y + h))

	# ########### YOUR CODE ENDS HERE ###########

	# # Return bounding box
	# return bounding_box

	# bounding_box = ((0,0),(0,0))

	# implementation = 0

	# # simplest implementation - find orange in the image
	# if implementation == 0:
	# 	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# 	height, width = hsv_img.shape[:2]
	# 	quarter_height = height // 4
	# 	start_row = height - 2*quarter_height  # 2nd quarter from bottom starts here
	# 	end_row = height - quarter_height
	# 	mask_line = np.zeros(img.shape[:2], np.uint8)
	# 	mask_line[start_row:end_row, :] = 255


	# 	masked_image = cv2.bitwise_and(hsv_img, hsv_img, mask=mask_line)

		
	# 	lower_orange = np.array([0, 50, 102]) #BEST 88221
	# 	upper_orange = np.array([15, 255, 255])
	# 	cropped = cv2.inRange(masked_image, lower_orange, upper_orange)
		
	# 	contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	# 	# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 	if contours:
	# 		largest_contour = max(contours, key=cv2.contourArea)
	# 		x, y, w, h = cv2.boundingRect(largest_contour)
	# 		bounding_box = ((x, y), (x + w, y + h))


	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box
