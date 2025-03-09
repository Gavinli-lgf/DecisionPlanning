import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

current_file_path = os.path.dirname(os.path.abspath(__file__))

# 参数设置
T = 10          # 总时间10s
tau = 0.5       # 时间步长0.5s 
N = int(T/tau)  # 总步数N=20
# tau = 1
# N = 5
Q = np.diag([0.0, 1.0, 2.0, 1.0, 2.0, 4.0])         # 状态误差权重(分别对一个x,y方向上的6个状态。Q[0][0]为0,说明cost中不关注x的位置)
R = np.diag([4.0, 4.0])                             # 控制输入权重(分别对应x,y方向上的输入jx,jy)
xF0 = np.array([2.0, 15.0, 0.0, 5.0, 0.0, 0.0])     # follower初始状态定义(x,vx,ax,y,vy,ay)
xFref = np.array([0.0, 15.0, 0.0, 5.0, 0.0, 0.0])   # follower目标状态(由Q[0][0]为0可知,不关注x状态,只关注另外5个状态)
# xL0 = np.array([12.0, 10.0, 0.0, 3.0, 0.0, 0.0])
xL0 = np.array([32.0, 10.0, 0.0, 3.0, 0.0, 0.0])    # leader初始状态定义
xLref = np.array([0.0, 10.0, 0.0, 5.0, 0.0, 0.0])   # leader目标状态(由Q[0][0]为0可知,不关注x状态,只关注另外5个状态)
addCollisionCons = True
vxRef = 15
# vxRef = 10

KF = 0.01       # folloer的cost系数(KF与KL和为1)
KL = 1 - KF     # leader的cost系数
# distF = 20    # collision ditance (conservative)
distF = 10      # collision ditance (agressive)(follower避障约束中要与leader的安全距离)
distL = 15      # leader避障约束中要与follower的安全距离
Kinfluence = 0  # leader对周围folloer影响cost的系数

# KF = 0.5
# KL = 1 - KF
# distF = 20    # collision ditance
# distL = 20
# Kinfluence = 1    # enable Jinfluence

# 状态转移矩阵(x,y方向都使用jerk的3阶模型)
A_np = np.array([[1.0, tau, 0.5*tau**2, 0.0, 0.0, 0.0],
                 [0.0, 1.0, tau, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, tau, 0.5*tau**2],
                 [0.0, 0.0, 0.0, 0.0, 1.0, tau],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# 控制输入矩阵
B_np = np.array([[1/6*tau**3, 0.0],
                 [0.5*tau**2, 0.0],
                 [tau, 0.0],
                 [0.0, 1/6*tau**3],
                 [0.0, 0.5*tau**2],
                 [0.0, tau]])

A = ca.MX(A_np)
B = ca.MX(B_np)

# 定义系统动力学(状态转移方程)
def dynamics(x, u):
    return ca.mtimes(A, x) + ca.mtimes(B, u)

# Function for Leader Follower Game
def gameLeaderFollower(xL0, xF0):

    # # 创建优化变量
    # XF = ca.MX.sym('XF', 6, N+1)
    # UF = ca.MX.sym('UF', 2, N)
    # XL = ca.MX.sym('XL', 6, N+1)
    # UL = ca.MX.sym('UL', 2, N)

    # 定义优化问题
    nx = 6  # 状态维度
    nu = 2  # 输入维度
    XF = ca.MX.sym('XF', nx, N+1)   # 6*21
    UF = ca.MX.sym('UF', nu, N)     # 2*20(follower的2个控制量jerk_f_x, jerk_f_y)
    XL = ca.MX.sym('XL', nx, N+1)   # 6*21
    UL = ca.MX.sym('UL', nu, N)     # 2*20(leader的2个控制量jerk_l_x, jerk_l_y)
    lenX = nx * (N+1)   # 6*21=121
    lenU = nu * N       # 2*20=40

    JF = 0          # cost:follower的代价函数
    JL = 0          # cost:leader的代价函数
    Jinfluence = 0  # cost:leader对周围follower的影响代价
    consF = []          # constrain: follower的等式约束
    consL = []          # constrain: leader的等式约束
    collisionConsF = [] # constrain: follower的不等式约束
    collisionConsL = [] # constrain: leader的不等式约束

    # 等式约束1:约束leader,follower的初始状态
    consF.append(XF[:, 0] - xF0)
    consL.append(XL[:, 0] - xL0)

    # 轨迹规划的目标
    for k in range(N):
        # cost
        xFrefCa = ca.MX(xFref)
        xLrefCa = ca.MX(xLref)
        JF += ca.mtimes([(XF[:, k+1] - xFrefCa).T, Q, (XF[:, k+1] - xFrefCa)]) + ca.mtimes([UF[:, k].T, R, UF[:, k]])
        JL += ca.mtimes([(XL[:, k+1] - xLrefCa).T, Q, (XL[:, k+1] - xLrefCa)]) + ca.mtimes([UL[:, k].T, R, UL[:, k]])
        Jinfluence += (XF[1, k+1] - vxRef) ** 2
        # constrain(等式约束2:约束leader,follower满足动力学方程)
        xF_next = dynamics(XF[:, k], UF[:, k])
        xL_next = dynamics(XL[:, k], UL[:, k])
        consF.append(XF[:, k+1] - xF_next)
        consL.append(XL[:, k+1] - xL_next)
        # collision(不等式约束: follower与leader分别的避障约束)
        if addCollisionCons:
            # px_L - px_F > dist  => dist + px_F - px_L < 0
            collisionConsF.append(distF + XF[0, k+1] - XL[0, k+1])
            collisionConsL.append(distL + XF[0, k+1] - XL[0, k+1])

    equConF = ca.vertcat(*consF)
    equConL = ca.vertcat(*consL)
    inequConsF = ca.vertcat(*collisionConsF)
    inequConsL = ca.vertcat(*collisionConsL)
    inequCon = ca.veccat(inequConsF, inequConsL)

    # define Lagrangian multipliers(即拉格朗日乘子lambda/mu的维度,分别等于对应等式/不等式约束的个数)
    lambda_ = ca.MX.sym('lambda', equConF.size1())
    mu = ca.MX.sym('nu', inequConsF.size1())

    # Lagrangian(构建follower的cost的拉格朗日函数)
    Lagrangian = JF + ca.mtimes([lambda_.T, equConF]) + ca.mtimes([mu.T, inequConsF])

    # KKT condition (equality)(构建follower的优化的kkt条件)
    grad_L_x = ca.gradient(Lagrangian, ca.vertcat(ca.reshape(XF, -1, 1), ca.reshape(UF, -1, 1)))
    complementary_slackness = ca.diag(mu) @ inequConsF
    equCon = ca.vertcat(equConF, equConL, grad_L_x, complementary_slackness)

    # construct the optimization problem
    x = ca.vertcat(ca.reshape(XF, -1, 1), ca.reshape(UF, -1, 1), ca.reshape(XL, -1, 1), ca.reshape(UL, -1, 1), lambda_, mu)
    lbx = ca.vertcat((2 * (lenU + lenX) + lambda_.size1()) * [-ca.inf] + mu.size1() * [0])
    nlp = {'x': x, 'f': KF * JF + KL * JL + Kinfluence * Jinfluence, 'g': ca.vertcat(equCon, inequCon)}

    # 设置求解器
    # opts = {'ipopt': {'print_level': 0}}
    # solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # 求解优化问题
    # sol = opti.solve()
    # x0_var = np.concatenate((np.reshape(x0, (-1, 1)), np.zeros((nx * N, 1))), axis=0)
    # u0_var = np.zeros((nu * N, 1))
    # init_guess = np.concatenate((x0_var, u0_var), axis=0)
    # sol = solver(x0=init_guess, lbx=-np.inf, ubx=np.inf, lbg=0, ubg=0)
    # without initial guess
    sol = solver(lbg=ca.DM(equCon.size1() * [0] + inequCon.size1() * [-ca.inf]), ubg=ca.DM((equCon.size1() + inequCon.size1()) * [0]), lbx = lbx)
    # sol = solver(lbg = 0, ubg = 0)

    # 提取解
    sol_x = sol['x'].full().flatten()
    XF_sol = sol_x[:lenX].reshape(N+1, nx).T
    UF_sol = sol_x[lenX:lenX+lenU].reshape(N, nu).T
    XL_sol = sol_x[lenX+lenU:2*lenX+lenU].reshape(N+1, nx).T
    UL_sol = sol_x[2*lenX+lenU:2*lenX+2*lenU].reshape(N, nu).T

    return XF_sol, UF_sol, XL_sol, UL_sol

XF_sol, UF_sol, XL_sol, UL_sol = gameLeaderFollower(xL0, xF0)

np.set_printoptions(precision=2)
# print('XF_sol')
# print(XF_sol)
# print('UF_sol')
# print(UF_sol)

# 可视化结果
time = np.arange(N+1)
plt.figure(figsize=(14, 8))

# px 轨迹
plt.subplot(4, 2, 1)
plt.plot(time, XF_sol[0, :], 'r-', label='px')
plt.plot(time, xFref[0]*np.ones(N+1), 'b--', label='xref px')
plt.title('Position in x')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()

# py 轨迹
plt.subplot(4, 2, 2)
plt.plot(time, XF_sol[3, :], 'r-', label='py')
plt.plot(time, xFref[3]*np.ones(N+1), 'b--', label='xref py')
plt.title('Position in y')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()

# # vx 轨迹
# plt.subplot(4, 2, 3)
# plt.plot(time, X_sol[1, :], 'r-', label='vx')
# plt.legend()

# # vy 轨迹
# plt.subplot(4, 2, 4)
# plt.plot(time, X_sol[4, :], 'r-', label='vy')
# plt.legend()

# # ax 轨迹
# plt.subplot(4, 2, 5)
# plt.plot(time, X_sol[2, :], 'r-', label='ax')
# plt.legend()

# # ay 轨迹
# plt.subplot(4, 2, 6)
# plt.plot(time, X_sol[5, :], 'r-', label='ay')
# plt.legend()

# # jx 轨迹
# plt.subplot(4, 2, 7)
# plt.plot(time[:-1], U_sol[0, :], 'g-', label='jx')
# plt.title('Jerk in x')
# plt.xlabel('Time step')
# plt.ylabel('Jerk')
# plt.legend()

# # jy 轨迹
# plt.subplot(4, 2, 8)
# plt.plot(time[:-1], U_sol[1, :], 'g-', label='jy')
# plt.title('Jerk in y')
# plt.xlabel('Time step')
# plt.ylabel('Jerk')
# plt.legend()

# y-x 轨迹
plt.figure(figsize=(7, 5))
plt.plot(XF_sol[0, :], XF_sol[3, :], 'r-', label='Trajectory')
plt.title('Trajectory (y vs. x)')
plt.xlabel('px')
plt.ylabel('py')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

# Create the animation
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

def update(frame):

    # clear plotting of last timestep
    ax[0].clear()
    ax[1].clear()
    
    # plot traj
    ax[0].plot(XF_sol[0, : frame+1], XF_sol[3, : frame+1], 'bo-', label='Traj_F')
    ax[0].plot(XL_sol[0, : frame+1], XL_sol[3, : frame+1], 'ro-', label='Traj_L')

    # plot collision distance
    ax[0].axvline(x = XF_sol[0, frame] + distF, color='b', linestyle='--', label='collisionF')
    ax[0].axvline(x = XL_sol[0, frame] - distL, color='r', linestyle='--', label='collisionL')

    # Add labels
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    # ax[0].legend()
    # ax[0].set_title('State Evolution')
    ax[0].set_title(f't = {frame * tau}')
    
    # Set limit
    ax[0].set_xlim(0, 160)
    ax[0].set_ylim(0, 10)
    ax[0].grid(True)

    # Plot 
    time = np.arange(0, frame * tau + tau, tau)
    ax[1].plot(time, XF_sol[1, : frame+1], 'bo-', label='vxF')
    ax[1].plot(time, XL_sol[1, : frame+1], 'ro-', label='vxL')
    
    # # Add labels
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('u')
    # # ax[1].legend()
    # # ax[1].set_title('Control Inputs')

    # # Set limit
    ax[1].set_xlim(0, T+2*tau)
    # ax[1].set_ylim(-0.2, 0.2)
    # ax[1].grid(True)

    # save fig
    plt.savefig(f'{current_file_path}/log/Game_{frame}.jpg')

ani = FuncAnimation(fig, update, frames=range(N+1), repeat=False)
ani.save(f'{current_file_path}/log/Game_animation.gif', writer='pillow')
ani.save(f'{current_file_path}/log/Game_animation.mp4', writer='ffmpeg')

# plt.show()