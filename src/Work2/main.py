import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 常量
WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 帧缓冲
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# ==============================
# De Casteljau 算法
# ==============================
def de_casteljau(points, t):
    p = np.array(points, dtype=np.float32)
    n = len(p) - 1
    for k in range(1, n + 1):
        for i in range(n - k + 1):
            p[i] = (1 - t) * p[i] + t * p[i + 1]
    return p[0]

# ==============================
# GPU 绘制内核
# ==============================
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        x, y = curve_points_field[i]
        px = int(x * WIDTH)
        py = int(y * HEIGHT)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = (0.0, 1.0, 0.0)

@ti.kernel
def clear_screen():
    for i, j in pixels:
        pixels[i, j] = (0.0, 0.0, 0.0)

# ==============================
# 主程序
# ==============================
def main():
    window = ti.ui.Window("Bezier Curve (De Casteljau)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []

    gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)

    while window.running:
        clear_screen()

        # 鼠标添加控制点
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(np.array(pos, dtype=np.float32))
            if window.event.key == 'c':
                control_points.clear()

        # 绘制曲线
        if len(control_points) >= 2:
            curve = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve[i] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 更新控制点（对象池）
        gui_points_np[:] = -10.0
        cnt = min(len(control_points), MAX_CONTROL_POINTS)
        if cnt > 0:
            gui_points_np[:cnt] = np.array(control_points)[:cnt]
        gui_points.from_numpy(gui_points_np)

        # 显示
        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=0.008, color=(1, 0, 0))

        window.show()

if __name__ == "__main__":
    main()