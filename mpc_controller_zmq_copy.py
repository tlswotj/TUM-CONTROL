"""
ZeroMQ-based implementation of the MPC controller originally written as a ROS 2 node.

This module removes all ROS 2 dependencies and instead uses a pair of ZeroMQ
publish/subscribe sockets to exchange data with a separate bridge process.  It
expects the following message types on its subscribed socket:

* ``{"type": "global_path", "pos_x": [...], "pos_y": [...], "ref_v": [...]}``
  A list of waypoints expressed as Cartesian ``x``/``y`` positions and a
  reference velocity for each point.  Upon reception the controller computes
  a corresponding list of yaw angles and stores the result for the MPC solver.

* ``{"type": "odom", "x": float, "y": float, "qx": float, "qy": float,
       "qz": float, "qw": float, "timestamp": float}``
  Odometry information describing the vehicle pose in the world.  The quaternion
  components ``qx``/``qy``/``qz``/``qw`` are converted to a yaw heading and
  finite differences are used to estimate longitudinal/lateral velocities,
  yaw rate and longitudinal acceleration.  The most recent odom sample is used
  as the current state for the MPC.

After processing both a global path and at least one odometry sample the
controller enters a periodic control loop (20 Hz by default) in which it solves
the nonlinear MPC problem and publishes a control command on its ZeroMQ PUB
socket as:

``{"type": "control", "speed": float, "steering_angle": float}``

A separate ROS 2 bridge is responsible for converting between ROS messages and
these JSON dictionaries.  See ``ros_zmq_bridge.py`` for details.
"""

import json
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq
import yaml

"""
The MPC controller depends on a `Nonlinear_Model_Predictive_Controller` class and
a `PlannerEmulator` helper.  In the original repository these live in
`Model_Predictive_Controller.Nominal_NMPC.NMPC_class` and
`Utils.MPC_sim_utils` respectively, but those packages may not be available
when running this script standalone.  To make the controller more robust,
we attempt to import from the original locations and fall back to local
modules if necessary.  See also the accompanying `utils.py` for
`PlannerEmulator`.
"""
try:
    # Prefer the original package structure if it exists.
    from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import (
        Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller,
    )
except ImportError:
    # Fall back to the local file `NMPC_class.py` when the package is not installed.
    from NMPC_class import (
        Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller,
    )

try:
    from Utils.MPC_sim_utils import PlannerEmulator
except ImportError:
    # Fall back to the local `MPC_sim_utils.py` when the package layout is absent.
    try:
        from MPC_sim_utils import PlannerEmulator  # type: ignore
    except ImportError:
        # Finally fall back to `utils.py` if neither exists.
        from utils import PlannerEmulator  # type: ignore

###############################################################################
# Helper functions for quaternion/yaw/angle handling.
###############################################################################

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Convert a quaternion into a yaw angle in radians.

    Parameters
    ----------
    x, y, z, w : float
        Quaternion components.

    Returns
    -------
    float
        Yaw angle in the range (-pi, pi].
    """
    # Standard Euler-from-quaternion conversion for yaw (Z axis).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_2pi(angle: float) -> float:
    """
    Wrap an angle to the range [0, 2*pi).

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Equivalent angle in the range [0, 2*pi).
    """
    twopi = 2.0 * math.pi
    return angle % twopi


def angle_diff(angle: float, reference: float) -> float:
    """
    Compute the signed difference between two angles.

    The result lies in the interval (-pi, pi].

    Parameters
    ----------
    angle : float
        Current angle.
    reference : float
        Reference angle to subtract.

    Returns
    -------
    float
        Difference ``angle - reference`` wrapped to (-pi, pi].
    """
    diff = angle - reference
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff <= -math.pi:
        diff += 2.0 * math.pi
    return diff


class MPCControllerZMQ:
    """
    A ZeroMQ-based MPC controller.  It closely follows the logic of the
    original ROS 2 ``MPC_controller_node`` but uses plain Python and ZeroMQ
    sockets for I/O.
    """

    def __init__(
        self,
        sub_address: str = "tcp://*:5555",
        pub_address: str = "tcp://*:5556",
        config_dir: str = "Config/",
        mpc_params_file: str = "EDGAR/MPC_params.yaml",
        sim_main_params_file: str = "EDGAR/sim_main_params.yaml",
        *,
        ref_v_unit: str = "mps",
        ref_v_min: float = 0.1,
        steering_max: float = 0.5,
        loop_circuit: bool = True,
    ) -> None:
        """
        Construct a new MPC controller that communicates over ZeroMQ.

        Parameters
        ----------
        sub_address : str, optional
            ZeroMQ address to bind the subscriber (incoming messages).  Default
            uses port 5555 on all interfaces.
        pub_address : str, optional
            ZeroMQ address to bind the publisher (control commands).  Default
            uses port 5556 on all interfaces.
        config_dir : str, optional
            Directory prefix for locating YAML configuration files.
        mpc_params_file : str, optional
            Name of the YAML file containing MPC parameters.
        sim_main_params_file : str, optional
            Name of the YAML file containing simulation parameters.
        """
        # Load YAML configuration for the MPC and main simulation.
        self.config_dir = config_dir
        params_path = f"{config_dir}{sim_main_params_file}"
        with open(params_path, "r") as file:
            sim_main_params = yaml.load(file, Loader=yaml.FullLoader)

        self.Tp = sim_main_params["Tp"]
        self.Ts_MPC = sim_main_params["Ts_MPC"]
        self.N = int(self.Tp / self.Ts_MPC)

        # Initial state [x, y, yaw, v_lon, v_lat, yaw_rate, delta_f, a_lon].
        self.current_pose = np.zeros(8, dtype=float)
        self.delta_f = 0.0

        # configuration for reference velocity handling
        # ref_v_unit specifies the unit of incoming reference velocities: "mps" or "kmh"
        # ref_v_min is the minimum positive velocity used to avoid division by zero in the planner
        self.ref_v_unit = ref_v_unit.lower()
        self.ref_v_min = ref_v_min

        # maximum steering angle (radians) used to saturate the integrated steering angle
        # if None, no saturation is applied
        self.steering_max = steering_max

        # whether the reference trajectory is treated as a closed loop
        self.loop_circuit = loop_circuit

        # Previous state for finite difference derivatives.
        self._prev_xy: Optional[Tuple[float, float]] = None
        self._prev_t: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._prev_v_lon: float = 0.0

        # Flags indicating whether initial data has been received.
        self._global_path_ready = False
        self._odom_ready = False

        # Storage for the reference trajectory.
        self.ref_traj: Dict[str, list] = {
            "pos_x": [],
            "pos_y": [],
            "ref_yaw": [],
            "ref_v": [],
        }

        # Construct the nonlinear MPC controller.
        mpc_params_path = f"{config_dir}{mpc_params_file}"
        self.mpc_params_file = mpc_params_file
        self.sim_main_params = sim_main_params


        # ZeroMQ context and sockets.
        self._context = zmq.Context()
        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.RCVHWM, 100000)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub_socket.bind(sub_address)

        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.SNDHWM, 100000)
        self._pub_socket.bind(pub_address)

        # Control loop timing.
        self._control_period = self.Ts_MPC
        self._next_control_time: float = time.monotonic()


    def _handle_global_path(self, msg: Dict[str, Any]) -> None:
        """
        Process a global path message and compute reference yaw, orientation and acceleration.
        Adjusts yaw to minimize wrap‑around and matches JSON yaw range.
        """
        px = msg.get("pos_x", [])
        py = msg.get("pos_y", [])
        pv = msg.get("ref_v", [])

        # convert velocities to m/s and clamp to minimum positive value
        converted_v = []
        for v in pv:
            try:
                v_float = float(v)
            except Exception:
                v_float = 0.0
            if self.ref_v_unit == "kmh":
                v_float = v_float / 3.6
            if v_float <= 0.0:
                v_float = self.ref_v_min
            converted_v.append(v_float)

        n = len(px)
        # initialise ref_traj with empty lists
        self.ref_traj = {
            "pos_x": list(px),
            "pos_y": list(py),
            "ref_v": converted_v,
            "ref_yaw": [],
            "orientation_x": [],
            "orientation_y": [],
            "orientation_z": [],
            "orientation_w": [],
            "ref_acc": []
        }

        # compute yaw for each segment
        for i in range(n - 1):
            dx = px[i + 1] - px[i]
            dy = py[i + 1] - py[i]
            self.ref_traj["ref_yaw"].append(math.atan2(dy, dx))

        # compute last yaw: if loop and last==first, reuse previous yaw to avoid wrap‑around
        if n > 0:
            if self.loop_circuit and n > 2:
                dx_last = px[0] - px[-1]
                dy_last = py[0] - py[-1]
                if abs(dx_last) < 1e-6 and abs(dy_last) < 1e-6:
                    yaw_last = self.ref_traj["ref_yaw"][-1]
                else:
                    yaw_last = math.atan2(dy_last, dx_last)
            elif n > 1:
                dx_last = px[-1] - px[-2]
                dy_last = py[-1] - py[-2]
                yaw_last = math.atan2(dy_last, dx_last)
            else:
                yaw_last = 0.0
            self.ref_traj["ref_yaw"].append(yaw_last)

        # unwrap yaw to reduce jumps, then wrap back to [-pi, pi]
        for i in range(1, len(self.ref_traj["ref_yaw"])):
            delta = self.ref_traj["ref_yaw"][i] - self.ref_traj["ref_yaw"][i - 1]
            if delta > math.pi:
                self.ref_traj["ref_yaw"][i] -= 2.0 * math.pi
            elif delta < -math.pi:
                self.ref_traj["ref_yaw"][i] += 2.0 * math.pi
        # wrap all yaw values into [-pi, pi] range
        for i in range(len(self.ref_traj["ref_yaw"])):
            y = self.ref_traj["ref_yaw"][i]
            y = (y + math.pi) % (2.0 * math.pi) - math.pi
            self.ref_traj["ref_yaw"][i] = y

        # compute orientation from yaw (x and y are always 0)
        for yaw in self.ref_traj["ref_yaw"]:
            self.ref_traj["orientation_x"].append(0.0)
            self.ref_traj["orientation_y"].append(0.0)
            self.ref_traj["orientation_z"].append(math.sin(yaw * 0.5))
            self.ref_traj["orientation_w"].append(math.cos(yaw * 0.5))

        # compute ref_acc for each segment
        for i in range(n - 1):
            dx = px[i + 1] - px[i]
            dy = py[i + 1] - py[i]
            dist = math.hypot(dx, dy)
            v1 = converted_v[i]
            v2 = converted_v[i + 1]
            avg_v = (v1 + v2) * 0.5 if (v1 + v2) > 0 else self.ref_v_min
            time = dist / avg_v if avg_v > 1e-6 else dist / self.ref_v_min
            acc = (v2 - v1) / max(time, 1e-6)
            self.ref_traj["ref_acc"].append(acc)

        # last acceleration: handle loop or repeat final segment
        if n > 0:
            if self.loop_circuit and n > 2:
                dx = px[0] - px[-1]
                dy = py[0] - py[-1]
                v1 = converted_v[-1]
                v2 = converted_v[0]
            elif n > 1:
                dx = px[-1] - px[-2]
                dy = py[-1] - py[-2]
                v1 = converted_v[-2]
                v2 = converted_v[-1]
            else:
                dx = dy = v1 = v2 = 0.0
            dist = math.hypot(dx, dy)
            avg_v = (v1 + v2) * 0.5 if (v1 + v2) > 0 else self.ref_v_min
            time = dist / avg_v if avg_v > 1e-6 else dist / self.ref_v_min
            acc_last = (v2 - v1) / max(time, 1e-6)
            self.ref_traj["ref_acc"].append(acc_last)

        # flag ready
        self._global_path_ready = True


    def _handle_odom(self, msg: Dict[str, Any]) -> None:
        """
        Process an odometry message and update the current pose and derived
        quantities (velocity, yaw rate, acceleration).

        Parameters
        ----------
        msg : dict
            Dictionary containing ``x``, ``y``, quaternion components and
            ``timestamp`` (seconds).
        """
        x = float(msg["x"])
        y = float(msg["y"])
        qx = float(msg["qx"])
        qy = float(msg["qy"])
        qz = float(msg["qz"])
        qw = float(msg["qw"])
        t = float(msg["timestamp"])

        # Yaw extraction and wrapping.
        yaw_raw = quat_to_yaw(qx, qy, qz, qw)  # (-pi, pi]
        yaw = wrap_to_2pi(yaw_raw)  # [0, 2pi)

        # Finite difference to compute world-frame velocities.
        if self._prev_xy is None or self._prev_t is None:
            dt = 0.0
            vx_w = 0.0
            vy_w = 0.0
        else:
            dt = max(t - self._prev_t, 0.0)
            if dt > 1e-6:
                dx = x - self._prev_xy[0]
                dy = y - self._prev_xy[1]
                vx_w = dx / dt
                vy_w = dy / dt
            else:
                vx_w = 0.0
                vy_w = 0.0

        # Rotate world-frame velocities into vehicle frame.
        c, s = math.cos(yaw), math.sin(yaw)
        v_lon = c * vx_w + s * vy_w
        v_lat = -s * vx_w + c * vy_w

        # Yaw rate.
        if self._prev_yaw is not None and dt > 1e-6:
            dyaw = angle_diff(yaw, self._prev_yaw)
            yaw_rate = dyaw / dt
        else:
            yaw_rate = 0.0

        # Longitudinal acceleration.
        if dt > 1e-6:
            a_lon = (v_lon - self._prev_v_lon) / dt
        else:
            a_lon = 0.0

        delta_f = self.delta_f

        # Update the current pose vector.
        self.current_pose = np.array(
            [x, y, yaw, v_lon, v_lat, yaw_rate, delta_f, a_lon], dtype=float
        )

        # Update previous state for next derivative.
        self._prev_xy = (x, y)
        self._prev_t = t
        self._prev_yaw = yaw
        self._prev_v_lon = v_lon

        self._odom_ready = True

    def _solve_mpc(self) -> Optional[Dict[str, float]]:
        """
        Solve the MPC problem using the most recent state and reference.

        Returns
        -------
        dict or None
            Dictionary containing ``speed`` and ``steering_angle`` if a valid
            control solution was found; otherwise ``None``.
        """
        # Generate a reference trajectory segment for the prediction horizon.
        # The PlannerEmulator returns the current index and a trimmed
        # trajectory of length N+1.
        current_ref_idx, current_ref_traj = PlannerEmulator(
            self.ref_traj, self.current_pose, self.N + 1, self.Tp, loop_circuit=self.loop_circuit
        )

        # Set the initial state for the MPC problem.

        # Solve the MPC.
        # self.MPC_time is unused here but retained for compatibility.
        try:
            u, pred_X, stats = self.MPC.solve(current_ref_traj)
        except Exception as e:
            print(f"[MPC] Exception during solve: {e}")
            return None
        print(f"[MPC] solved")
        # stats[-1] holds the acados return status; 0 indicates success.
        if isinstance(stats, (list, tuple)) and len(stats) > 0:
            status = stats[-1]
        else:
            status = 0

        if status != 0:
            print(f"[MPC] acados returned status {status}")
            # Attempt to reinitialize the solver with the current state.
            try:
                self.MPC.reintialize_solver(self.current_pose)
                print(f"[MPC] current_pose x= {self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, yaw={self.current_pose[2]:.2f}, v_lon={self.current_pose[3]:.2f}")
            except Exception as e:
                print(f"[MPC] Failed to reinitialize solver: {e}")
            return None
        next_x = pred_X[1,:]
        #self.MPC.set_initial_state(self.current_pose)
        print(f"[MPC] Odom received: x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, yaw={self.current_pose[2]:.2f}, v_lon={self.current_pose[3]:.2f}")
        print(f"[MPC] Predicted next state: x={next_x[0]:.2f}, y={next_x[1]:.2f}, yaw={next_x[2]:.2f}, v_lon={next_x[3]:.2f}")
        self.MPC.set_initial_state(next_x)

        # The MPC returns two control values: the longitudinal jerk (rate of change
        # of acceleration) and the front steering rate.  The original code
        # incorrectly interpreted the jerk as an instantaneous speed command.
        # To generate a meaningful speed command we integrate the jerk twice:
        # first to update the longitudinal acceleration and then again to
        # update the velocity.  This preserves the physical meaning of the
        # control input.
        try:
            jerk = float(u[0])
            steering_rate = float(u[1])
        except Exception:
            return None

        # Current longitudinal acceleration and velocity from the state vector.
        a_lon_current = float(self.current_pose[7]) if len(self.current_pose) > 7 else 0.0
        v_lon_current = float(self.current_pose[3]) if len(self.current_pose) > 3 else 0.0

        # Update longitudinal acceleration by integrating jerk over one MPC step.
        a_lon_next = a_lon_current + jerk * self.Ts_MPC
        # Update velocity by integrating the updated acceleration over one step.
        v_cmd = v_lon_current + a_lon_next * self.Ts_MPC

        # Integrate steering rate to obtain the front steering angle.
        self.delta_f += steering_rate * self.Ts_MPC
        # saturate the steering angle if a limit is specified
        if self.steering_max is not None:
            if self.delta_f > self.steering_max:
                self.delta_f = self.steering_max
            elif self.delta_f < -self.steering_max:
                self.delta_f = -self.steering_max

        return {"speed": v_cmd, "steering_angle": self.delta_f}

    def run(self) -> None:
        """
        Main event loop.  Processes incoming messages and executes the control
        loop at the prescribed rate.
        """
        
        print("[MPC] Waiting for both global path and odometry...")
        poller = zmq.Poller()
        poller.register(self._sub_socket, zmq.POLLIN)

        # Block until both a global path and an odometry sample have been received.
        while not (self._global_path_ready and self._odom_ready):
            socks = dict(poller.poll(timeout=100))
            if self._sub_socket in socks and socks[self._sub_socket] == zmq.POLLIN:
                try:
                    data_bytes = self._sub_socket.recv()
                    msg = json.loads(data_bytes.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(msg, dict) and "type" in msg:
                    if msg["type"] == "global_path":
                        self._handle_global_path(msg)
                        print(f"[MPC] Global path received with {len(self.ref_traj['pos_x'])} points.")
                    elif msg["type"] == "odom":
                        self._handle_odom(msg)
                        #print(f"[MPC] Odom received: x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, yaw={self.current_pose[2]:.2f}, v_lon={self.current_pose[3]:.2f}")
        X0_MPC = self.current_pose  # initial state
        print(f"[MPC] Initial pose: x={X0_MPC[0]:.2f}, y={X0_MPC[1]:.2f}, yaw={X0_MPC[2]:.2f}, v_lon={X0_MPC[3]:.2f}")
        self.MPC = Model_Predictive_Controller(
            self.config_dir, self.mpc_params_file, self.sim_main_params, X0_MPC
        )
        print("[MPC] Initial data received.  Entering control loop.")

        # Start periodic control loop.
        self.MPC.reintialize_solver(self.current_pose)
        while True:
            now = time.monotonic()
            # Process all available incoming messages.
            while True:
                try:
                    data_bytes = self._sub_socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    msg = json.loads(data_bytes.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(msg, dict) or "type" not in msg:
                    continue
                if msg["type"] == "global_path":
                    self._handle_global_path(msg)
                    print(f"[MPC] Global path received with {len(self.ref_traj['pos_x'])} points.")
                elif msg["type"] == "odom":
                    self._handle_odom(msg)
                    print(f"[MPC] Odom received: x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, yaw={self.current_pose[2]:.2f}, v_lon={self.current_pose[3]:.2f}")

            # Time management for fixed-rate control execution.

            if now >= self._next_control_time:
                # Update target time for the next cycle.
                self._next_control_time = now + self._control_period

                if self._global_path_ready and self._odom_ready:
                    cmd = self._solve_mpc()
                    if cmd is not None:
                        out = {"type": "control", "speed": cmd["speed"], "steering_angle": cmd["steering_angle"]}
                        try:
                            self._pub_socket.send_string(json.dumps(out))
                        except Exception as e:
                            print(f"[MPC] Failed to publish control command: {e}")

            # Sleep briefly to avoid busy-waiting.
            time.sleep(0.001)


def main() -> None:
    """
    Standalone entry point for running the ZeroMQ MPC controller.
    """
    controller = MPCControllerZMQ()
    controller.run()


if __name__ == "__main__":
    main()