"""
ROS 2 to ZeroMQ bridge for the MPC controller.

This node subscribes to ROS 2 topics (``/global_path`` and ``/odom``) and
publishes a drive command to ``/cmd_drive``.  Incoming ROS messages are
converted to simple JSON dictionaries and sent over ZeroMQ to the MPC
controller.  Control commands produced by the MPC are received via ZeroMQ
and translated back into ``AckermannDriveStamped`` messages for ROS.

The bridge assumes the MPC controller binds a ZeroMQ SUB socket on
``tcp://localhost:5555`` and a PUB socket on ``tcp://localhost:5556``.  These
addresses may be customised via the ``sub_address`` and ``pub_address``
parameters when constructing ``ZMQBridgeNode``.

Note that this bridge intentionally ignores any ``/local_path`` messages
because the original MPC example did not implement local path handling.
"""

import json
import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# Visualisation of predicted trajectories requires Marker and MarkerArray from
# visualization_msgs and the Point type from geometry_msgs.
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import zmq


class ZMQBridgeNode(Node):
    """
    Bridge ROS 2 topics to and from a ZeroMQ-based MPC controller.

    The node subscribes to ``/global_path`` and ``/odom`` and forwards these
    messages as JSON to the controller.  It concurrently listens on a ZeroMQ
    SUB socket for control commands and publishes them to ``/cmd_drive``.
    """

    def __init__(self, sub_address: str = "tcp://localhost:5555", pub_address: str = "tcp://localhost:5556") -> None:
        super().__init__("zmq_bridge_node")
        self.get_logger().info("Initialising ZeroMQ bridge node")

        # ZeroMQ context and sockets.  The PUB socket forwards ROS data to the
        # MPC controller and the SUB socket receives control commands.
        self._context = zmq.Context()
        # Publisher: forward ROS messages to the MPC
        self._pub = self._context.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 100000)
        try:
            self._pub.connect(sub_address)
        except Exception as e:
            self.get_logger().error(f"Failed to connect PUB socket to {sub_address}: {e}")

        # Subscriber: receive control commands from the MPC
        self._sub = self._context.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, 100000)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            self._sub.connect(pub_address)
        except Exception as e:
            self.get_logger().error(f"Failed to connect SUB socket to {pub_address}: {e}")

        # ROS publishers and subscribers
        cmd_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._cmd_pub = self.create_publisher(AckermannDriveStamped, "/cmd_drive", cmd_qos)

        # Publisher for the predicted MPC trajectory.  The MPC controller
        # includes the predicted x/y positions in its control message.  These
        # coordinates are visualised as a line strip using a MarkerArray on
        # the '/mpc_predicted_path' topic.
        marker_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._marker_pub = self.create_publisher(MarkerArray, "/mpc_predicted_path", marker_qos)
        self._ref_traj_pub = self.create_publisher(MarkerArray, "/mpc_ref_path", marker_qos)

        # Subscribe to global_path (transient/local)
        global_path_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Path, "/global_path", self._global_path_callback, global_path_qos)

        # Subscribe to odom (default reliability)
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

        # Thread for listening to control commands from ZeroMQ
        self._shutdown = False
        self._listener_thread = threading.Thread(target=self._control_listener, daemon=True)
        self._listener_thread.start()

    # ------------------------------------------------------------------
    # ROS subscription callbacks
    # ------------------------------------------------------------------
    def _global_path_callback(self, msg: Path) -> None:
        """Forward the received global path message to the MPC controller via ZeroMQ."""
        # Extract lists of x, y positions and reference velocities.  In this
        # convention the ``z`` component of the position encodes the desired
        # speed at that waypoint.
        pos_x = []
        pos_y = []
        ref_v = []
        for pose in msg.poses:
            pos_x.append(pose.pose.position.x)
            pos_y.append(pose.pose.position.y)
            # The reference velocity is stored in the z component of the
            # position field for convenience.
            ref_v.append(pose.pose.position.z)
        data = {"type": "global_path", "pos_x": pos_x, "pos_y": pos_y, "ref_v": ref_v}
        try:
            self._pub.send_string(json.dumps(data))
        except Exception as e:
            self.get_logger().warn(f"Failed to publish global path over ZMQ: {e}")
        self.get_logger().info(f"Published global path with {len(pos_x)} points")

    def _odom_callback(self, msg: Odometry) -> None:
        """Forward the received odometry message to the MPC controller via ZeroMQ."""
        # Flatten quaternion into individual components.
        q = msg.pose.pose.orientation
        # Time stamp in seconds with sub‑second precision
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        data = {
            "type": "odom",
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "qx": q.x,
            "qy": q.y,
            "qz": q.z,
            "qw": q.w,
            "timestamp": t,
        }
        try:
            self._pub.send_string(json.dumps(data))
        except Exception as e:
            self.get_logger().warn(f"Failed to publish odom over ZMQ: {e}")
        #self.get_logger().info(f"Published odom at time {t:.3f}, x={data['x']:.2f}, y={data['y']:.2f}")

    # ------------------------------------------------------------------
    # ZMQ listener thread
    # ------------------------------------------------------------------
    def _control_listener(self) -> None:
        """Listen for control messages from the MPC and publish to ROS."""
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        while not self._shutdown and rclpy.ok():
            socks = dict(poller.poll(timeout=100))
            if self._sub in socks and socks[self._sub] == zmq.POLLIN:
                try:
                    msg_bytes = self._sub.recv()
                    cmd = json.loads(msg_bytes.decode("utf-8"))
                except Exception as e:
                    self.get_logger().warn(f"Failed to decode control message: {e}")
                    continue
                # Expect a dict with type ``control``
                if isinstance(cmd, dict) and cmd.get("type") == "control":
                    try:
                        speed = float(cmd.get("speed", 0.0))
                        steering_angle = float(cmd.get("steering_angle", 0.0))
                    except Exception:
                        self.get_logger().warn("Invalid control fields in message")
                        continue
                    # Formulate a ROS AckermannDriveStamped message
                    ack_msg = AckermannDriveStamped()
                    ack_msg.header.stamp = self.get_clock().now().to_msg()
                    ack_msg.header.frame_id = "base_link"
                    ack_msg.drive.speed = speed
                    ack_msg.drive.steering_angle = steering_angle
                    self._cmd_pub.publish(ack_msg)
                    self.get_logger().info(f"Published control: speed={speed:.2f}, steering_angle={steering_angle:.2f}")

                    # If the control message includes predicted x/y coordinates
                    # (sent by the MPC controller) then convert them into a
                    # MarkerArray for visualisation.  The expected fields are
                    # 'pred_x' and 'pred_y', both lists of equal length.  When
                    # present, a single Marker of type LINE_STRIP is created.
                    pred_x = cmd.get("pred_x")
                    pred_y = cmd.get("pred_y")
                    ref_x = cmd.get("ref_x")
                    ref_y = cmd.get("ref_y")
                    if isinstance(pred_x, list) and isinstance(pred_y, list) and len(pred_x) == len(pred_y) and len(pred_x) > 0:
                        marker_array = MarkerArray()
                        marker = Marker()
                        marker.header.stamp = self.get_clock().now().to_msg()
                        # Use map frame for the trajectory visualisation; this
                        # should match the frame_id of the global path used by
                        # the planner.  Adjust if necessary for your setup.
                        marker.header.frame_id = "map"
                        marker.ns = "mpc_predicted_path"
                        marker.id = 0
                        marker.type = Marker.LINE_STRIP
                        marker.action = Marker.ADD
                        # Populate the points for the line strip
                        marker.points = []
                        for x_val, y_val in zip(pred_x, pred_y):
                            p = Point()
                            try:
                                p.x = float(x_val)
                                p.y = float(y_val)
                            except Exception:
                                continue
                            p.z = 0.0
                            marker.points.append(p)
                        # Set a reasonable scale for the line thickness
                        marker.scale.x = 0.05
                        marker.scale.y = 0.0
                        marker.scale.z = 0.0
                        # Set colour: blue with full opacity
                        marker.color.r = 0.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0
                        marker.color.a = 1.0
                        # No orientation needed for a line strip
                        marker.pose.orientation.w = 1.0
                        # Add the marker to the array and publish
                        marker_array.markers.append(marker)
                        self._marker_pub.publish(marker_array)
                        self.get_logger().info(f"Published predicted path with {len(marker.points)} points")
                    if isinstance(ref_x, list) and isinstance(ref_y, list) and len(ref_x) == len(ref_y) and len(ref_x) > 0:
                        marker_array = MarkerArray()
                        marker = Marker()
                        marker.header.stamp = self.get_clock().now().to_msg()
                        # Use map frame for the trajectory visualisation; this
                        # should match the frame_id of the global path used by
                        # the planner.  Adjust if necessary for your setup.
                        marker.header.frame_id = "map"
                        marker.ns = "mpc_ref_path"
                        marker.id = 0
                        marker.type = Marker.LINE_STRIP
                        marker.action = Marker.ADD
                        # Populate the points for the line strip
                        marker.points = []
                        for x_val, y_val in zip(ref_x, ref_y):
                            p = Point()
                            try:
                                p.x = float(x_val)
                                p.y = float(y_val)
                            except Exception:
                                continue
                            p.z = 0.0
                            marker.points.append(p)
                        # Set a reasonable scale for the line thickness
                        marker.scale.x = 0.05
                        marker.scale.y = 0.0
                        marker.scale.z = 0.0
                        # Set colour: red with full opacity
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                        # No orientation needed for a line strip
                        marker.pose.orientation.w = 1.0
                        # Add the marker to the array and publish
                        marker_array.markers.append(marker)
                        self._ref_traj_pub.publish(marker_array)
                        self.get_logger().info(f"Published reference path with {len(marker.points)} points")
        # Clean up sockets once the loop exits
        self._sub.close()
        self._pub.close()
        self._context.term()

    # ------------------------------------------------------------------
    # Shutdown handling
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        """Override destroy_node to shut down the listener thread cleanly."""
        self._shutdown = True
        # Allow some time for the poller loop to exit
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
        super().destroy_node()


def main(args: Any = None) -> None:
    """
    Entry point to run the ROS 2 ZeroMQ bridge.

    Use environment variables or command line parameters to customise the
    ZeroMQ addresses if necessary.
    """
    rclpy.init(args=args)
    # Optionally parse arguments for custom ZMQ addresses here.
    node = ZMQBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()