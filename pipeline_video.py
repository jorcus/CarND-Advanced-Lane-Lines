from moviepy.editor import VideoFileClip
from line import Line
from helper import *
from image_processing import combined_threshold
from pipeline_image import pipeline
import calibrate_camera

if __name__ == "__main__":
    left_lane = Line()
    right_lane = Line()
    white_output = 'project_video_out.mp4'  
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
