import taichi as ti

ti.init(arch=ti.cuda)

# ==================== 模拟参数配置 ====================
N = 8                           # 更小的网格以提高性能
GRID_SIZE = 0.3                 # 更大的网格间距
dt = 0.005                      # 更大的时间步长
gravity = ti.Vector([0.0, -9.8, 0.0])
ks_structure = 300.0            # 降低劲度系数
kd = 5                        # 降低阻尼系数
mass_value = 1.0
max_velocity = 25.0

# ==================== 数据结构定义 ====================
TOTAL_POINTS = N * N

pos = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_POINTS)
vel = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_POINTS)

# 线条数据（存储在GPU上）
NUM_LINES = (N-1)*N + N*(N-1)
line_verts = ti.Vector.field(3, dtype=ti.f32, shape=NUM_LINES * 2)

paused = ti.field(dtype=ti.i32, shape=())
method = ti.field(dtype=ti.i32, shape=())

# ==================== 初始化函数 ====================
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        pos[idx] = ti.Vector([
            (i - N / 2) * GRID_SIZE + 1.0,  # 往右挪1米
            2.0,
            (j - N / 2) * GRID_SIZE
        ])
        vel[idx] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def init_lines():
    line_idx = 0
    
    for i in range(N):
        for j in range(N - 1):
            idx1 = i * N + j
            idx2 = i * N + j + 1
            line_verts[line_idx * 2] = pos[idx1]
            line_verts[line_idx * 2 + 1] = pos[idx2]
            line_idx += 1
    
    for j in range(N):
        for i in range(N - 1):
            idx1 = i * N + j
            idx2 = (i + 1) * N + j
            line_verts[line_idx * 2] = pos[idx1]
            line_verts[line_idx * 2 + 1] = pos[idx2]
            line_idx += 1

# ==================== 更新线条（GPU内完成，无数据传输） ====================
@ti.kernel
def update_lines():
    line_idx = 0
    
    for i in range(N):
        for j in range(N - 1):
            idx1 = i * N + j
            idx2 = i * N + j + 1
            line_verts[line_idx * 2] = pos[idx1]
            line_verts[line_idx * 2 + 1] = pos[idx2]
            line_idx += 1
    
    for j in range(N):
        for i in range(N - 1):
            idx1 = i * N + j
            idx2 = (i + 1) * N + j
            line_verts[line_idx * 2] = pos[idx1]
            line_verts[line_idx * 2 + 1] = pos[idx2]
            line_idx += 1

# ==================== 数值积分求解器 ====================
@ti.kernel
def step_explicit():
    for i in range(TOTAL_POINTS):
        if i != 0 and i != N - 1:
            force = mass_value * gravity - kd * vel[i]
            
            # 水平邻居
            if i % N != 0:
                j_idx = i - 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if (i + 1) % N != 0:
                j_idx = i + 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            # 垂直邻居
            if i >= N:
                j_idx = i - N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if i < TOTAL_POINTS - N:
                j_idx = i + N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            acc = force / mass_value
            pos[i] = pos[i] + vel[i] * dt
            vel[i] = vel[i] + acc * dt
            
            speed = vel[i].norm()
            if speed > max_velocity:
                vel[i] = vel[i] * (max_velocity / speed)

@ti.kernel
def step_semi_implicit():
    for i in range(TOTAL_POINTS):
        if i != 0 and i != N - 1:
            force = mass_value * gravity - kd * vel[i]
            
            if i % N != 0:
                j_idx = i - 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if (i + 1) % N != 0:
                j_idx = i + 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if i >= N:
                j_idx = i - N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if i < TOTAL_POINTS - N:
                j_idx = i + N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            acc = force / mass_value
            vel[i] = vel[i] + acc * dt
            
            speed = vel[i].norm()
            if speed > max_velocity:
                vel[i] = vel[i] * (max_velocity / speed)
            
            pos[i] = pos[i] + vel[i] * dt

@ti.kernel
def step_implicit_iter():
    for i in range(TOTAL_POINTS):
        if i != 0 and i != N - 1:
            force = mass_value * gravity - kd * vel[i]
            
            if i % N != 0:
                j_idx = i - 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if (i + 1) % N != 0:
                j_idx = i + 1
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if i >= N:
                j_idx = i - N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            if i < TOTAL_POINTS - N:
                j_idx = i + N
                diff = pos[i] - pos[j_idx]
                dist = diff.norm()
                if dist > 1e-6:
                    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
            
            acc = force / mass_value
            vel[i] = vel[i] + acc * dt
            
            speed = vel[i].norm()
            if speed > max_velocity:
                vel[i] = vel[i] * (max_velocity / speed)
            
            pos[i] = pos[i] + vel[i] * dt

def reset_cloth():
    init_positions()
    init_lines()
    paused[None] = 0
    method[None] = 0

# ==================== 主函数 ====================
def main():
    paused[None] = 0
    method[None] = 0
    
    init_positions()
    init_lines()
    
    window = ti.ui.Window("Mass-Spring Net", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()
    
    camera.position(0.0, 2.5, 4.0)
    camera.lookat(0.0, 1.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    
    while window.running:
        gui.begin("Control", 0.02, 0.02, 0.16, 0.22)
        
        if gui.button("Explicit"):
            method[None] = 0
        if gui.button("Semi-Impl"):
            method[None] = 1
        if gui.button("Implicit"):
            method[None] = 2
        
        gui.text(f"Method: {['Exp', 'Semi', 'Imp'][method[None]]}")
        
        gui.text("---")
        
        if gui.button("Pause"):
            paused[None] = 1 - paused[None]
        if gui.button("Reset"):
            reset_cloth()
        
        gui.end()
        
        if not paused[None]:
            if method[None] == 0:
                step_explicit()
            elif method[None] == 1:
                step_semi_implicit()
            else:
                step_implicit_iter()
            
            update_lines()
        
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.0, 5.0, 0.0), color=(1.0, 1.0, 1.0))
        
        # 使用GPU数据直接渲染，无需CPU传输
        scene.lines(line_verts, color=(0.8, 0.6, 0.4), width=2.0)
        scene.particles(pos, radius=0.025, color=(0.7, 0.5, 0.3))
        
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
