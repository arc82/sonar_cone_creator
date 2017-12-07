from __future__ import division
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline 
import cv2
import time
import pickle


class radial_point_calculator:
    m_points = [] 
    m_points_initialized = False
    m_num_beams = 0
    m_num_lines = 0
    m_size = 0
    m_angle = 0

    def get_points(self, num_beams, num_lines, size, angle):
        if not(self.m_num_beams == num_beams and self.m_num_lines == num_lines and self.m_size == size and self.m_angle == angle and self.m_points_initialized):
            print 'Calculating new x y points' 
            for beam in range(num_beams):
                for line in range(num_lines):
                    r = (line/num_lines) * size
                    theta = (beam/num_beams) * np.deg2rad(angle)

                    x = r * np.cos(theta)
                    y = r * np.sin(theta)

                    self.m_points.append([x,y])

            self.m_points = np.array(self.m_points)
            self.m_points_initialized = True
            self.m_num_beams = num_beams
            self.m_num_lines = num_lines
            self.m_size = size
            self.m_angle = angle
        
        return self.m_points
        


def show_image():
    # Read sample data
    image = cv2.imread('sample_data.png', 0)
    pickle_data = pickle.load( open("/home/andrew/Desktop/data3.pkl", "rb"))
    count = 0;
    radial_point_calculator_obj = radial_point_calculator()

    for i in range(0, len(pickle_data)):
        raw_sonar_image = pickle_data[i]
        cone_image = get_radial_image(raw_sonar_image, 300, 120, radial_point_calculator_obj)

        # Rotate image
        m = cv2.getRotationMatrix2D((0,0), 135, 1)
        width = int(300*np.sqrt(2))
        t = np.array([[0, 0, width//2],[0, 0, 300]])
        rotated_image = cv2.warpAffine(cone_image, m+t, (width, 300))

        # Display image
        cv2.imshow('Cone Image', rotated_image)
        cv2.waitKey(1)

def get_radial_image(data_in, size, angle, radial_point_calculator_obj):
    print 'Get radial image started'
    if (angle < 1 or angle > 180):
        raise ValueError('Angle must be between 1 and 180')

    num_lines = data_in.shape[0]
    num_beams = data_in.shape[1]

    # cartesian co-ords for each beam/line & its value
    values = []
    points = radial_point_calculator_obj.get_points(num_beams, num_lines, size, angle) 

    for beam in range(num_beams):
        for line in range(num_lines):
            v = data_in[line, beam]

            values.append(v)

    values = np.array(values)

    x_grid, y_grid = np.mgrid[0:size, 0:size]
    return griddata(points, values, (x_grid, y_grid), method='nearest') / 255

if __name__ == '__main__':
    print("Started sonar_cone_test")
    show_image()
