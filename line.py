import numpy as np
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.detected = False  # line detected in the last iteration
        self.recent_xfitted = [] # x values of the last n fits of the line
        self.bestx = None # average x values of the fitted line over the last n iterations
        self.best_fit = None #polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])] #polynomial coefficients for the most recent fit
        self.radius_of_curvature = None #radius of curvature of the line in some units
        self.line_base_pos = None #distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float') #difference in fit coefficients between last and new fits
        self.allx = None #x values for detected line pixels
        self.ally = None #y values for detected line pixels

