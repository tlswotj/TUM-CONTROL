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