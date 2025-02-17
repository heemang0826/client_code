import pyrealsense2 as rs
import numpy as np


SENSOR_DEPTH    = 0
SENSOR_COLOR    = 1
SENSOR_GYRO     = 1
SENSOR_ACCEL    = 0
VIDEO           = 0
IMU             = 1

CLIPPING_DISTANCE_IN_METERS = 3

class SensorRealsense:
    def __init__(self, mode=VIDEO, clipping_distance_in_meters=CLIPPING_DISTANCE_IN_METERS):
        '''
        VIDEO   == 0
        IMU     == 1
        '''
        
        self.is_video = mode

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        if self.is_video == VIDEO:
            found_rgb = False
            for s in self.device.sensors:
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    found_rgb = True
                    break
            if not found_rgb:
                print("The Demo Requires Depth Camera with color sensor")
                raise("The Demo Requires Depth Camera with color sensor")
            
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)

        self.profile = self.pipeline.start(self.config)

        if self.is_video == VIDEO:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            self.clipping_distance = clipping_distance_in_meters / depth_scale

            align_to = rs.stream.color
            self.align = rs.align(align_to)

        print("Realsense Init Done.")

    def get_video_from_pipeline(self):
        if self.is_video != VIDEO:
            print("this instance isn't Video pipeline.")
            return -1
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_timestamp = aligned_depth_frame.get_timestamp()
                    
        color_image = np.asanyarray(color_frame.get_data())
        color_timestamp = color_frame.get_timestamp()
        
        return (depth_image, depth_timestamp), (color_image, color_timestamp)
    