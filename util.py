import numpy as np
import math

# projected distance of a point3 from point1 in the direction of point1 to point2
def projected_distance(point1, point2, point3):
    '''
    point1: (x,y) coordinates of the first point
    point2: (x,y) coordinates of the second point
    point3: (x,y) coordinates of the car
    '''
    # get the vector from point1 to point2
    vector1 = np.array(point2) - np.array(point1)
    # get the vector from point1 to point3
    vector2 = np.array(point3) - np.array(point1)
    # get the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    # get the magnitude of the vector
    magnitude = np.linalg.norm(vector1)
    # get the projected distance
    projected_distance = dot_product/magnitude
    return projected_distance

# Define a function that takes a three points and find distance of the first point from the line passing through the other two points
def vertical_distance_from_two_points(point1, point2, point3):
  # Unpack the tuple and assign the coordinates to variables
  (x, y), (x1, y1), (x2, y2) = point1, point2, point3
  # Check if x2 and x1 are the same
  if x2 == x1:
    # The line is vertical and its equation is y = y1
    # The vertical distance is the difference between the y-coordinates
    d = abs(y - y1)
    # print("vertical","distance",d)
  else:
    # Calculate the slope of the line passing through (x1,y1) and (x2,y2)
    m = (y2 - y1) / (x2 - x1)
    # Calculate the y-intercept of the line
    b = y1 - m * x1
    # Calculate the perpendicular distance from (x,y) to the line using the formula
    d = abs((m * x - y + b) / math.sqrt(m ** 2 + 1))
    # print("diagonal","distance",d)
  # Return the vertical distance
  return d
