import numpy as np
from scipy.interpolate import griddata
import cv2
import time

def show_image():
    # Read sample data
    image = cv2.imread('sample_data.png', 0)
    cone_image = get_radial_image(image, 300, 90)

    # Rotate image
    m = cv2.getRotationMatrix2D((0,0), 135, 1)
    width = int(300*np.sqrt(2))
    t = np.array([[0, 0, width//2],[0, 0, 300]])
    rotated_image = cv2.warpAffine(cone_image, m+t, (width, 300))

    # Display image
    cv2.imshow('Cone Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def get_radial_image(data_in, size, angle):

    if (angle < 1 or angle > 180):
        raise ValueError('Angle must be between 1 and 180')

    num_lines = data_in.shape[0]
    num_beams = data_in.shape[1]

    # cartesian co-ords for each beam/line & its value
    points = []
    values = []

    for beam in range(num_beams):
        for line in range(num_lines):
            r = (line/num_lines) * size
            theta = (beam/num_beams) * np.deg2rad(angle)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            v = data_in[line, beam]

            points.append([x,y])
            values.append(v)

    points = np.array(points)
    values = np.array(values)

    x_grid, y_grid = np.mgrid[0:size, 0:size]
    return griddata(points, values, (x_grid, y_grid), method='linear') / 255

if __name__ == '__main__':
    print("Started sonar_cone_test")
    show_image()
