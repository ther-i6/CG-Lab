import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

def de_casteljau(points, t):
    p = np.array(points, dtype=np.float32)
    n = len(p) - 1
    for k in range(1, n + 1):
        for i in range(n - k + 1):
            p[i] = (1 - t) * p[i] + t * p[i + 1]
    return p[0]

def cubic_bspline(points, t):
    n = len(points) - 1
    if n < 3:
        return np.array([0.0, 0.0], dtype=np.float32)
    
    segment = min(int(t * (n - 2)), n - 3)
    local_t = t * (n - 2) - segment
    
    basis_matrix = np.array([
        [-1/6, 3/6, -3/6, 1/6],
        [3/6, -6/6, 3/6, 0/6],
        [-3/6, 0/6, 3/6, 0/6],
        [1/6, 4/6, 1/6, 0/6]
    ], dtype=np.float32)
    
    p0 = points[segment]
    p1 = points[segment + 1]
    p2 = points[segment + 2]
    p3 = points[segment + 3]
    
    t_vec = np.array([local_t**3, local_t**2, local_t, 1.0], dtype=np.float32)
    
    x = t_vec @ basis_matrix @ np.array([p0[0], p1[0], p2[0], p3[0]])
    y = t_vec @ basis_matrix @ np.array([p0[1], p1[1], p2[1], p3[1]])
    
    return np.array([x, y], dtype=np.float32)

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        x, y = curve_points_field[i]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = int(x * WIDTH) + dx
                py = int(y * HEIGHT) + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pixel_center_x = (px + 0.5) / WIDTH
                    pixel_center_y = (py + 0.5) / HEIGHT
                    distance = ti.sqrt((x - pixel_center_x) ** 2 + (y - pixel_center_y) ** 2)
                    weight = ti.max(0.0, 1.0 - distance * 200.0)
                    pixels[px, py] += weight * ti.Vector([0.0, 1.0, 0.0])
                    pixels[px, py] = ti.min(pixels[px, py], ti.Vector([1.0, 1.0, 1.0]))

@ti.kernel
def clear_screen():
    for i, j in pixels:
        pixels[i, j] = (0.0, 0.0, 0.0)

def main():
    window = ti.ui.Window("Bezier Curve & B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    use_bspline = False

    gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)

    while window.running:
        clear_screen()

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(np.array(pos, dtype=np.float32))
            if window.event.key == 'c':
                control_points.clear()
            if window.event.key == 'b':
                use_bspline = not use_bspline

        if len(control_points) >= 2:
            curve = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                if use_bspline and len(control_points) >= 4:
                    curve[i] = cubic_bspline(control_points, t)
                else:
                    curve[i] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        gui_points_np[:] = -10.0
        cnt = min(len(control_points), MAX_CONTROL_POINTS)
        if cnt > 0:
            gui_points_np[:cnt] = np.array(control_points)[:cnt]
        gui_points.from_numpy(gui_points_np)

        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=0.008, color=(1, 0, 0))
        window.show()

if __name__ == "__main__":
    main()