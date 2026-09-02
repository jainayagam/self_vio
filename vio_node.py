import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


class vio_node(Node): 

    def __init__(self):

        super().__init__('vio_node')


        self.K = np.array([[343.5, 0, 320.0], [0, 343.5, 240.0], [0, 0, 1.0]])
        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None 

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = R.identity() #identity matix


    
        self.last_imu_time = None
        self.last_frame_time = None
        self.imu_accel_buffer = []

        
        self.create_subscription(Image, '/iris/camera', self.image_cb, 10)
        self.create_subscription(Imu, '/iris/imu', self.imu_callback, 50)
        self.odom_pub = self.create_publisher(Odometry, '/simple_vio/odometry', 10)

        self.feature_params = dict(maxCorners=200, qualityLevel=0.01,
                                    minDistance=10, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.create_timer(1.0, self.timer_callback)



    def timer_callback(self): 
        self.get_logger().info("looping")



    def imu_callback(self , msg: Imu): 
        t = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9

        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        gravity_world = np.array([0,0, 9.81])

        gravity_body = self.orientation.inv().apply(gravity_world)

        accel_free = accel - gravity_body

        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        if self.last_imu_time is not None: 
            dt = t - self.last_imu_time

            if dt>0: 
                delta_rot = R.from_rotvec(gyro*dt)
                self.orientation = self.orientation * delta_rot
                self.imu_accel_buffer.append((t, accel_free, dt))

        self.last_imu_time = t



    def estimate_translation_distance(self): 
        if not self.imu_accel_buffer: 
            return 0.0
        v = self.velocity.copy()
        displacement = np.zeros(3)

        for (_, accel_free, dt) in self.imu_accel_buffer: 
            world_accel = self.orientation.apply(accel_free)

            displacement += v*dt + 0.5 * world_accel * dt**2
            v += world_accel*dt


        self.velocity = v * 0
        self.imu_accel_buffer.clear()

        return np.linalg.norm(displacement)

   #def image_callback(self, msg: Image):

    def image_cb(self, msg: Image):

        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:

            self.prev_gray = gray

            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return


        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK( self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 20:
            self.prev_gray = gray

            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return

        E, mask = cv2.findEssentialMat(good_curr, good_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)


        if E is None:
            return

        
        _, Rmat, t_unit, mask = cv2.recoverPose(E, good_curr, good_prev, self.K)

    
        scale = self.estimate_translation_distance()

        if scale < 1e-4:
            scale = 0.02  

        t_metric = (t_unit.flatten() / np.linalg.norm(t_unit)) * scale

        world_delta = self.orientation.apply(t_metric)
        self.position += world_delta

        self.publish_odometry(msg.header.stamp)
        
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)

    def publish_odometry(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])
        q = self.orientation.as_quat()
        odom.pose.pose.orientation.x = 1 #q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self.odom_pub.publish(odom)


def main():


    rclpy.init()
    node = vio_node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()


