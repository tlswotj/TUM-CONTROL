import time
import numpy as np
from casadi import *
from scipy.spatial.transform import Rotation as R
from Utils.MPC_sim_utils import *
from Utils.Logging_Plotting import Logger
import yaml
from utils import *
from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller
import rclpy
from rclpy.task import Future
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped


class MPC_controller_node(Node):
    def __init__(self):

        #Loading parameters
        super(). __init__('mpc_controller_node')
        config_path, logs_path  = 'Config/', 'Logs/'
        MPC_params_file         = "EDGAR/MPC_params.yaml" 
        sim_main_params_file    = "EDGAR/sim_main_params.yaml"
        with open(config_path + sim_main_params_file, 'r') as file:
            sim_main_params = yaml.load(file, Loader=yaml.FullLoader)

        #Subscribers
        global_path_qos_profile = QoSProfile( depth=10, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.global_path_subscription = self.create_subscription(Path, '/global_path', self.global_path_callback, global_path_qos_profile)
        self.global_path_ready = Future()
        self.local_path_subscription = self.create_subscription(Path, '/local_path', self.local_path_callback, 0)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 0)
        self.odom_ready = Future()

        self.get_logger().info("waiting for global")
        rclpy.spin_until_future_complete(self, self.global_path_ready)
        self.get_logger().info("global path received!")
        rclpy.spin_until_future_complete(self, self.odom_ready)
        self.get_logger().info("odom received!")

        #Initializations
        self.Tp      = sim_main_params['Tp']
        self.Ts_MPC  = sim_main_params['Ts_MPC']
        self.N       = int(self.Tp / self.Ts_MPC)
        self.delta_f = 0.0 #조향각 초기화
        self.current_pose = np.zeros(8, dtype=float)
        # 이전 샘플 저장
        self._prev_xy    = None   # (x, y)
        self._prev_t     = None   # float sec
        self._prev_yaw   = None   # rad
        self._prev_v_lon = 0.0    # m/s

        X0_MPC = self.current_pose # initial state MPC: [x,y,yaw,v_lon,v_lat,yaw_rate,delta_f,acceleration]

        self.MPC = Model_Predictive_Controller(config_path, MPC_params_file, sim_main_params, X0_MPC)

    def global_path_callback(self, msg=Path()):
        self.ref_traj = {'pos_x':[], 'pos_y':[], 'ref_yaw':[], 'ref_v':[]}
        for pose in msg.poses:
            self.ref_traj['pos_x'].append(pose.pose.position.x)
            self.ref_traj['pos_y'].append(pose.pose.position.y)
            self.ref_traj['ref_v'].append(pose.pose.position.z)
        n = len(self.ref_traj['pos_x'])
        for i in range(n - 1):
            dx = self.ref_traj['pos_x'][i+1] - self.ref_traj['pos_x'][i]
            dy = self.ref_traj['pos_y'][i+1] - self.ref_traj['pos_y'][i]
            yaw = np.arctan2(dy, dx)
            self.ref_traj['ref_yaw'].append(yaw)
        # 마지막 yaw: 마지막 점과 첫 번째 점을 연결해서 계산
        dx = self.ref_traj['pos_x'][0] - self.ref_traj['pos_x'][-1]
        dy = self.ref_traj['pos_y'][0] - self.ref_traj['pos_y'][-1]
        yaw_last = np.arctan2(dy, dx)
        self.ref_traj['ref_yaw'].append(yaw_last)
        self.global_path_ready.set_result(True)
    
    def local_path_callback(self, msg=Path()):
        self.local_traj = {'pos_x':[], 'pos_y':[], 'ref_yaw':[], 'ref_v':[]}
        for pose in msg.poses:
            self.local_traj['pos_x'].append(pose.pose.position.x)
            self.local_traj['pos_y'].append(pose.pose.position.y)
            self.local_traj['ref_yaw'].append(0)
            self.local_traj['ref_v'].append(pose.pose.position.z)
    

    def odom_callback(self, msg=Odometry()):
# 위치
        # 1) 위치, yaw
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw_raw = quat_to_yaw(q.x, q.y, q.z, q.w)  # (-pi, pi]
        yaw = wrap_to_2pi(yaw_raw)                 # [0, 2pi)

        # 2) 시간 (sec + nsec)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 3) Δt 및 월드 속도 (차분 기반)
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

        # 4) 월드 → 차량 좌표로 속도 회전
        c, s = math.cos(yaw), math.sin(yaw)
        v_lon =  c * vx_w + s * vy_w
        v_lat = -s * vx_w + c * vy_w

        # 5) yaw_rate (각도 래핑 고려)
        if self._prev_yaw is not None and dt > 1e-6:
            dyaw = angle_diff(yaw, self._prev_yaw)  # (-pi, pi]
            yaw_rate = dyaw / dt
        else:
            yaw_rate = 0.0

        # 6) 종방향 가속도 (a_lon)
        if dt > 1e-6:
            a_lon = (v_lon - self._prev_v_lon) / dt
        else:
            a_lon = 0.0

        # 7) 조향각 delta_f (자전거 모델 근사; 저속시 0)

        delta_f = self.delta_f

        # 8) 결과 업데이트
        self.current_pose = np.array(
            [x, y, yaw, v_lon, v_lat, yaw_rate, delta_f, a_lon],
            dtype=float
        )

        # 9) 이전 상태 갱신
        self._prev_xy    = (x, y)
        self._prev_t     = t
        self._prev_yaw   = yaw
        self._prev_v_lon = v_lon
    
        if not self.odom_ready.done():
            self.odom_ready.set_result(True)

    def MPC_solve(self):
        current_ref_idx, current_ref_traj = PlannerEmulator(self.ref_traj, self.current_pose, self.N+1, self.Tp, loop_circuit=True)
        self.MPC.set_initial_state(self.current_pose)
        self.MPC_time = self._prev_t
        self.u, self.pred_X, self.MPC_stats = self.MPC.solve(current_ref_traj)
        if self.MPC_stats[-1] != 0:
            print("acados returned status {} in closed loop iteration {}.".format(MPC_stats[-1], i))
            #################################################################
            self.MPC.reintialize_solver(self.current_pose)
    
    def publish_control(self, pub):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.delta_f += (float(self.u[1])*self.Ts_MPC)
        msg.drive.steering_angle = self.delta_f  # 조향각
        msg.drive.speed = float(self.u[0])           # 속도
        pub.publish(msg)
        
        

def run_controller():
    rclpy.init()
    node = MPC_controller_node()

    # 제어 명령 퍼블리셔 (토픽명은 환경에 맞게 변경 가능)
    cmd_qos = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE
    )
    pub = node.create_publisher(AckermannDriveStamped, '/cmd_drive', cmd_qos)

    rate_hz = 20.0
    period = 1.0 / rate_hz
    next_t = time.monotonic()

    try:
        while rclpy.ok():
            # 콜백 처리(구독 메시지 수신)
            rclpy.spin_once(node, timeout_sec=0.0)

            # 순서 보장: 1) MPC 계산 → 2) 명령 퍼블리시
            node.MPC_solve()
            node.publish_control(pub)

            # 20 Hz 유지
            next_t += period
            sleep_dt = next_t - time.monotonic()
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)
            else:
                # 주기가 밀리면 기준 재설정
                next_t = time.monotonic()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    # 기존 spin 기반 main 대신 실행 함수 호출
    run_controller()


if __name__ == '__main__':
    main()