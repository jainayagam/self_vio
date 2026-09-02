import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
import pymap3d as pm  


class gps_local_node(Node):

    def __init__(self):
        super().__init__('gps_local_node')

        self.origin_set = False
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        self.create_subscription(NavSatFix, '/iris/gps', self.gps_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, '/gps_local/odometry', 10)

    def gps_cb(self, msg: NavSatFix):
        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude


        if not self.origin_set:
            self.origin_lat = lat
            self.origin_lon = lon
            self.origin_alt = alt
            self.origin_set = True
            self.get_logger().info(
                f"GPS origin set: {lat:.7f}, {lon:.7f}, {alt:.2f}")


        e, n, u = pm.geodetic2enu(
            lat, lon, alt,
            self.origin_lat, self.origin_lon, self.origin_alt
        )

        self.publish_odometry(msg.header.stamp, e, n, u)

    def publish_odometry(self, stamp, x, y, z):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)

        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = gps_local_node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()