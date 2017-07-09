#!/usr/bin/env python
from __future__ import division
import numpy as np
import math
import cv2
import time


def show_image():
	#image = np.loadtxt('/home/cauv/Desktop/test/sonar_sample_data2.txt')
	image = cv2.imread('sample_data.png', 0)
	# print(image)
	# res = cv2.resize(image, (128*5, 64*5), interpolation=cv2.INTER_LINEAR)
	#res = cv2.resize(image, (128*5, 64*5), interpolation=cv2.INTER_LINEAR)

	cone_image = get_radial_image(image, 300, 90)

	
	cv2.imshow('Original data', image)
	cv2.imshow('Cone image', cone_image)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)

def get_radial_image(data_in, pixel_dimension, angle):
	"""data_in is the numpy array from a np.reshape command on the iamgeData from the sonar, pixel_dimension is the dimension of the square image that will contain the cone (which is a slice of a circle), angle is the angle of the cone produced (in degrees).
Centre of cone with be at the bottom of the image i.e. the last row but in the middle column
Use trig to find which beam a given pixel in the output image is in.
Use pythagerous to the centre of the cone to find out what line the pixel is.
"""
	print("Started get_radial_image")	
	print("want a cone of angle ", angle, " and image of dimension ", pixel_dimension)	
	
	if (angle < 1 or angle > 180):
		print("Must input an angle between 1 and 180 into get_radial_image() function")
		return -1


	numLines = data_in.shape[0] #Assuming that the number of lines is the number of rows
	numBeams = data_in.shape[1] #Assuming that the number of beams is the number of columns
	
	output = np.zeros((pixel_dimension, pixel_dimension))

	#With x, y coordinates being 0,0 at the top left of the image
	cone_origin_x = pixel_dimension / 2
	cone_origin_y = pixel_dimension

	cone_radius = pixel_dimension / 2
	

	for y in range(0, pixel_dimension):
		for x in range(0, pixel_dimension):
			#Now access the array with output[y,x] 
			dx = x - cone_origin_x #This makes dx negative when it is to the 
                                   #left of the cone_origin and positive when
                                   #it is to the right
			dy = cone_origin_y - y #This makes dy always positive
			angle_from_vertical = np.arctan(dx/dy) #Angle from vertical will be 
												   #between -pi/2 and +pi/2
			angle_from_vertical = 180 * (angle_from_vertical/math.pi) 
			#Angle between -90 and 90
			
			if (angle_from_vertical < angle/(0-2) or angle_from_vertical > angle/2):
				#This pixel is not in any of the beams so leave this element at 0
				output[y,x] = 0
			else:
				dis_to_origin = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

				if dis_to_origin < cone_radius:

					#Now find which beam it belongs to
					#If there were 50 beams and the cone needed to be 100 degrees 
					#then each beam would be 2 degrees
					beam_angle_step = angle / numBeams
					#print("Beam angle step is ", beam_angle_step)
					#Say current angle is -40 degrees, then you are 10 degrees into 
					#the cone so you are the 10/2 = 5th beam in
					beam_num = (angle_from_vertical + angle/2) / beam_angle_step
					
					
					
					
					
					#If the radius was 200 and it was 100 away from the origin, then
					#it would be half way through the lines.
					#This assumes that data points far away from the sonar are at 
					#the end of the lines list for each beam
					#
					#numLines - 1 since it's indexed from 0
					line_num = (dis_to_origin / (pixel_dimension / 2)) * (numLines - 1)

					#Now round the line and beam nums
					beam_num = round(beam_num)
					line_num = round(line_num)

					if beam_num == numBeams:
						beam_num = beam_num - 1
						# This is a stop gap measure to prevent out of array bounds errors
						#Need to think of better solution


					#print("beam-num is ", beam_num)
					#print("line-num is ", line_num)
					#print("numLines is ", numLines)
					#print("numBeams is ", numBeams)

					#Divide by 255 since when numpy array is converted to an image, 0 - 1 in the 
					#numpy array is mapped to 0-255 in the cv image so all numpy values must be
					#between 0 and 1
					output[y,x] = data_in[line_num, beam_num] / 255 

					
				else:
					output[y,x] = 0

	return output

def get_test_image():
	output = np.zeros((300, 300))
	for y in range(300):
		for x in range(300):
			output[y,x] = (x%255)/255
			
	
	return output
if __name__ == '__main__':
	print("Started sonar_cone_test")
	show_image()
