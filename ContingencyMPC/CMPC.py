import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

current_file_path = os.path.dirname(os.path.abspath(__file__))

# Parameters:可以测试如下4组参数，分别对应CMPC的2种工况，RMPC的2种工况。

# # 1. CMPC - obstacle pops out
cmpc = True     # True 使用CMPC; False 使用RMPC.
obs_pop = True  # True obs跳出; False obs不跳出.
# Pc = 0.25     # Cost weight for contingency control(也是Contingency planning的概率)
Pc = 0.7

# # 2. CMPC - obstacle does not pop out
# cmpc = True
# obs_pop = False
# # Pc = 1e-2       # for unique minimum solution
# Pc = 0.3

# # 3. RMPC - obstacle pops out
# cmpc = False
# obs_pop = True
# Pc = 1.0 - 1e-2   

# # 4. RMPC - obstacle does not pop out
# cmpc = False
# obs_pop = False
# Pc = 1.0 - 1e-2      

# Define the horizon(这里的N可以理解为t)
N = 12      # 总仿真12步(生成12帧动画)
N_obs = 10  # N for obstacle (obs所处的x向位置.即当时间单位为1时，该位置就等于步数值)
N_c = 4     # N for contingency (pops out) obs在第4个时间步时开始跳出
# N_o = N_obs - N_c   # N for observe (pops out)

Pn = 1 - Pc     # Cost weight for normal control(nominal planning的概率)
# y_obs_max = 1.0   # Obstacle position maximum
y_obs_max = 0.5     # Obstacle position maximum (for N_obs = 10) obs的y向最大位置到0.5处
y_obs_0 = -1.0      # obs的y向初始位置是-1.0处
# obs_speed = 0.25  # Obstacle speed
obs_speed = 0.2  # Obstacle speed   obs的y向速度为0.2(0.2个单位/仿真步长)
obs_width = 0.4 # Obstacle width
collision_width = 0.1 # Collision width

# Initial conditions for receding horizional plannnig (nominal planning和contigency planning的初始状态)
x0 = 0
y0n = 0
y0c = 0

# Function to perform contingency MPC(输入: 当前时刻t,nominal与contingency的初始状态)
def solve_Cmpc(x0, y0n, y0c, t):

    # Define the control inputs (定义控制输入)
    un = ca.MX.sym('un', N)
    uc = ca.MX.sym('uc', N)

    # Define the states (定义状态)
    x = ca.MX.zeros(N+1)
    yn = ca.MX.zeros(N+1)
    yc = ca.MX.zeros(N+1)
    x[0] = x0
    yn[0] = y0n
    yc[0] = y0c

    # State update equations (使用状态转移方程，更新状态)
    for k in range(N):
        x[k+1] = x[k] + 1
        yn[k+1] = yn[k] + un[k]
        yc[k+1] = yc[k] + uc[k]

    # Define the cost function (定义成本函数)
    J = 0
    for k in range(N):             
        J += Pn * un[k]**2 + Pc * uc[k]**2

    # Define the constraints (定义约束条件:控制输入约束)
    g = []      # 状态约束g：y位置的约束
    lbg = []    # 输入约束lbg,ubg(lbg 都取 0; ubg 初始取0，其他都取 ca.inf)
    ubg = []

    # Control constraint
    g.append(un[0] - uc[0])
    lbg.append(0)
    ubg.append(0)
    
    # Obstacle avoidance constraint (障碍物避障约束)
    if t <= N_c or not obs_pop:
        y_obs = y_obs_0
    else:        
        y_obs = obs_speed * (t - N_c) + y_obs_0  
    # y_obs_N = min(y_obs_max, obs_speed * (N_obs - t) + y_obs)
    y_obs_N = obs_speed * (N_obs - t) + y_obs
        
    for k in range(N+1):
        if k + t >= N_obs:

            # if contingency occur
            if t > N_c and obs_pop:
                # both see the obstacle
                g.append(yc[k] - y_obs_N)
                lbg.append(0)
                ubg.append(ca.inf)
                g.append(yn[k] - y_obs_N)
                lbg.append(0)
                ubg.append(ca.inf)
            
            else:
                # only yc see the obstacle
                g.append(yc[k] - y_obs_N - obs_width)
                lbg.append(0)
                ubg.append(ca.inf)
                # g.append(y_obs_N - yc[k])
                # lbg.append(0)
                # ubg.append(ca.inf)
                

    # Convert the problem to a QP
    qp = {'x': ca.vertcat(un, uc), 'f': J, 'g': ca.vertcat(*g)}
    S = ca.qpsol('S', 'osqp', qp)

    # Solve the problem
    sol = S(lbg=lbg, ubg=ubg)
    u_opt = sol['x']

    # Extract the optimal control inputs and states
    un_opt = u_opt[:N].full().flatten()
    uc_opt = u_opt[N:2*N].full().flatten()
    x_opt = np.zeros(N+1)
    yn_opt = np.zeros(N+1)
    yc_opt = np.zeros(N+1)
    x_opt[0] = x0
    yn_opt[0] = y0n
    yc_opt[0] = y0c

    for k in range(N):
        x_opt[k+1] = x_opt[k] + 1
        yn_opt[k+1] = yn_opt[k] + un_opt[k]
        yc_opt[k+1] = yc_opt[k] + uc_opt[k]

    return x_opt, yn_opt, yc_opt, un_opt, uc_opt, y_obs, y_obs_N

# Initialize lists to store the results for animation
x_hist = []     #记录 mpc x结果序列的历史值(即list[list[]])
xR = []         #   即list[],只记录历史的 x[0]
yn_hist = []    #记录 nominal mpc 求出的y向位置的历史结果 (即list[list[]])
yc_hist = []    #记录 contingency mpc 求出的y向位置的历史结果 (即list[list[]])
yR = []         #   记录融合 nominal 与 contingency 之后的y向历史位置 (即list[],只记录历史的y[0])
un_hist = []    #记录 nominal mpc 求出的un的历史结果 (即list[list[]])
uc_hist = []    #记录 contingency mpc 求出的uc的历史结果 (即list[list[]])
uR = []         #   记录融合 nominal 与 contingency 之后实际输入的历史值(对应MPC，每次取求出的u的序列的初始值u[0])
y_obs_hist = []     # for t
y_obs_N_hist = []   # for N_obs

# Initial conditions for rolling horizon
x_curr = x0
y0n_curr = y0n
y0c_curr = y0c

# Perform rolling horizon MPC
for t in range(N):

    # solve MPC on timestep t
    x_opt, yn_opt, yc_opt, un_opt, uc_opt, y_obs, y_obs_N = solve_Cmpc(x_curr, y0n_curr, y0c_curr, t)
    # print(f"t = {t}, y_obs = {y_obs}")
    x_hist.append(x_opt)
    yn_hist.append(yn_opt)
    yc_hist.append(yc_opt)
    un_hist.append(un_opt)
    uc_hist.append(uc_opt)
    y_obs_hist.append(y_obs)
    y_obs_N_hist.append(y_obs_N)
    uR.append(uc_opt[0])
    xR.append(x_curr)
    yR.append(y0c_curr)    # yn_opt[1] = yc_opt[1] since un_opt[0] = uc_opt[0]
    
    # Update initial conditions for the next step(注：这里应该用状态转移方程来求下一个状态的，但是作者偷懒了。)
    x_curr = x_opt[1]
    y0n_curr = yn_opt[1]
    y0c_curr = yc_opt[1]
    

# Create the animation
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

"""
动画的核心，它负责绘制每一帧的内容。(参数 frame, 表示当前帧的索引。)
"""
def update(frame):

    # clear plotting of last timestep
    ax[0].clear()
    ax[1].clear()
    
    # Plot planning states
    if cmpc:
        ax[0].plot(x_hist[frame], yn_hist[frame], 'bo', label='yn')
    ax[0].plot(x_hist[frame], yc_hist[frame], 'r^', label='yc')

    # Plot history
    ax[0].plot(xR[:frame+1], yR[:frame+1], 'm--', label='yR')

    # Plot obstacle
    y_min = -1.2
    rect_y_obs = Rectangle((N_obs - obs_width / 2, y_min), obs_width, y_obs_hist[frame] - y_min, linewidth=1, edgecolor='r', facecolor='r', alpha = 1)
    rect_y_obs_N = Rectangle((N_obs - obs_width / 2, y_min), obs_width, y_obs_N_hist[frame] - y_min, linewidth=1, edgecolor='r', facecolor='r', alpha = 0.3)
    ax[0].add_patch(rect_y_obs)
    ax[0].add_patch(rect_y_obs_N)
    # ax[0].axhspan(-1, 1, xmin=0.8, xmax=1.0, color='red', alpha=0.5, label='Actual Obstacle')

    # Plot contingency 
    if obs_pop and frame >= N_c:
        ax[0].axvline(x=4, color='r', linestyle='--', label='Contingency')

    # Add labels
    # ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    # ax[0].legend()
    # ax[0].set_title('State Evolution')
    ax[0].set_title(f'k = {frame}')
    
    # Set limit
    ax[0].set_xlim(0, 12)
    ax[0].set_ylim(y_min, -y_min)
    ax[0].grid(True)

    # Plot control inputs (number of state = number of control + 1)
    if cmpc:
        ax[1].step(x_hist[frame][:-1], un_hist[frame], 'bo', where='post', label='un')
    ax[1].step(x_hist[frame][:-1], uc_hist[frame], 'r^', where='post', label='uc')

    # Plot contingency 
    if obs_pop and frame >= N_c:
        ax[1].axvline(x=4, color='r', linestyle='--', label='Contingency')

    # Plot history
    ax[1].plot(xR[:frame+1], uR[:frame+1], 'm--', label='uR')
    
    # Add labels
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u')
    # ax[1].legend()
    # ax[1].set_title('Control Inputs')

    # Set limit
    ax[1].set_xlim(0, 12)
    ax[1].set_ylim(-0.2, 0.2)
    ax[1].grid(True)

    # save fig
    plt.savefig(f'{current_file_path}/log/frame_{frame}.jpg')

# matplotlib.animation 模块中的一个类，用于创建基于函数的动画。它通过重复调用用户定义的函数来生成动画的每一帧
ani = FuncAnimation(fig, update, frames=range(N), repeat=False)
ani.save(f'{current_file_path}/log/mpc_animation.gif', writer='pillow')
ani.save(f'{current_file_path}/log/mpc_animation.mp4', writer='ffmpeg')
# plt.show()
