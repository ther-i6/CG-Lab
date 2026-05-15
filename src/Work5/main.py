import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu, offline_cache=False)

res = (640, 480)

light_pos = ti.Vector.field(3, dtype=float, shape=())
light_pos[None] = tm.vec3([2.0, 3.0, 4.0])
light_color = tm.vec3(1.0, 1.0, 1.0)
bg_color = tm.vec3(0.0, 0.0, 0.0)

max_bounces = ti.field(dtype=int, shape=())
max_bounces[None] = 3

epsilon = 1e-4
mirror_reflectivity = 0.8

red_sphere_center = tm.vec3(-1.5, 0.0, 0.0)
red_sphere_radius = 1.0
red_sphere_color = tm.vec3(0.8, 0.1, 0.1)

silver_sphere_center = tm.vec3(1.5, 0.0, 0.0)
silver_sphere_radius = 1.0
silver_sphere_color = tm.vec3(0.9, 0.9, 0.9)

ground_y = -1.0
ground_normal = tm.vec3(0.0, 1.0, 0.0)

img = ti.Vector.field(3, dtype=float, shape=res)

MATERIAL_DIFFUSE = 0
MATERIAL_MIRROR = 1
MATERIAL_GROUND = 2

@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    hit = 0
    t = -1.0
    hit_pos = tm.vec3(0.0)
    normal = tm.vec3(0.0)
    material_id = MATERIAL_DIFFUSE

    oc = ray_origin - center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    if discriminant > 0:
        t0 = (-b - tm.sqrt(discriminant)) / (2.0 * a)
        t1 = (-b + tm.sqrt(discriminant)) / (2.0 * a)
        if t0 > epsilon:
            t = t0
        else:
            if t1 > epsilon:
                t = t1

    if t > epsilon:
        hit_pos = ray_origin + t * ray_dir
        normal = tm.normalize(hit_pos - center)
        hit = 1

    return hit, t, hit_pos, normal, material_id

@ti.func
def intersect_ground(ray_origin, ray_dir):
    hit = 0
    t = -1.0
    hit_pos = tm.vec3(0.0)
    normal = tm.vec3(0.0, 1.0, 0.0)
    material_id = MATERIAL_GROUND

    if abs(ray_dir.y) > epsilon:
        t = (ground_y - ray_origin.y) / ray_dir.y
        if t > epsilon:
            hit_pos = ray_origin + t * ray_dir
            hit = 1

    return hit, t, hit_pos, normal, material_id

@ti.func
def check_shadow(hit_pos, normal):
    shadow_ray_origin = hit_pos + normal * epsilon
    shadow_ray_dir = tm.normalize(light_pos[None] - shadow_ray_origin)

    shadow_hit = 0

    hit_red, t_red, pos_red, norm_red, mat_red = intersect_sphere(
        shadow_ray_origin, shadow_ray_dir, red_sphere_center, red_sphere_radius)
    hit_silver, t_silver, pos_silver, norm_silver, mat_silver = intersect_sphere(
        shadow_ray_origin, shadow_ray_dir, silver_sphere_center, silver_sphere_radius)
    hit_ground, t_ground, pos_ground, norm_ground, mat_ground = intersect_ground(
        shadow_ray_origin, shadow_ray_dir)

    min_t = 1e10

    if hit_red == 1 and t_red > epsilon and t_red < min_t:
        min_t = t_red
        shadow_hit = 1

    if hit_silver == 1 and t_silver > epsilon and t_silver < min_t:
        min_t = t_silver
        shadow_hit = 1

    if hit_ground == 1 and t_ground > epsilon and t_ground < min_t:
        min_t = t_ground
        shadow_hit = 1

    return shadow_hit

@ti.func
def compute_diffuse_color(hit_pos, normal, object_color):
    ambient = 0.2 * light_color * object_color

    shadow_hit = check_shadow(hit_pos, normal)

    in_shadow = 1
    if shadow_hit == 0:
        in_shadow = 0

    result_color = ambient

    if shadow_hit == 0:
        L = tm.normalize(light_pos[None] - hit_pos)
        diffuse_factor = tm.max(0.0, tm.dot(normal, L))
        diffuse = 0.7 * diffuse_factor * light_color * object_color
        result_color = ambient + diffuse

    return result_color

@ti.func
def get_checkerboard_color(hit_pos):
    checker = int(ti.floor(hit_pos.x) + ti.floor(hit_pos.z)) % 2
    color = tm.vec3(0.1, 0.1, 0.1)
    if checker == 0:
        color = tm.vec3(1.0, 1.0, 1.0)
    return color

@ti.func
def trace_ray(ray_origin, ray_dir):
    final_color = tm.vec3(0.0)
    throughput = tm.vec3(1.0)

    for bounce in range(max_bounces[None]):
        hit_red, t_red, pos_red, norm_red, mat_red = intersect_sphere(
            ray_origin, ray_dir, red_sphere_center, red_sphere_radius)
        hit_silver, t_silver, pos_silver, norm_silver, mat_silver = intersect_sphere(
            ray_origin, ray_dir, silver_sphere_center, silver_sphere_radius)
        hit_ground, t_ground, pos_ground, norm_ground, mat_ground = intersect_ground(
            ray_origin, ray_dir)

        hit = 0
        min_t = 1e10
        hit_pos = tm.vec3(0.0)
        normal = tm.vec3(0.0)
        material_id = MATERIAL_DIFFUSE
        object_color = tm.vec3(0.0)

        if hit_red == 1 and t_red > epsilon and t_red < min_t:
            min_t = t_red
            hit = 1
            hit_pos = pos_red
            normal = norm_red
            material_id = MATERIAL_DIFFUSE
            object_color = red_sphere_color

        if hit_silver == 1 and t_silver > epsilon and t_silver < min_t:
            min_t = t_silver
            hit = 1
            hit_pos = pos_silver
            normal = norm_silver
            material_id = MATERIAL_MIRROR
            object_color = silver_sphere_color

        if hit_ground == 1 and t_ground > epsilon and t_ground < min_t:
            min_t = t_ground
            hit = 1
            hit_pos = pos_ground
            normal = norm_ground
            material_id = MATERIAL_GROUND
            object_color = get_checkerboard_color(hit_pos)

        if hit == 0:
            break

        if material_id == MATERIAL_DIFFUSE or material_id == MATERIAL_GROUND:
            color = compute_diffuse_color(hit_pos, normal, object_color)
            final_color += throughput * color
            break

        elif material_id == MATERIAL_MIRROR:
            reflected_dir = ray_dir - 2.0 * tm.dot(ray_dir, normal) * normal
            reflected_dir = tm.normalize(reflected_dir)
            ray_origin = hit_pos + normal * epsilon
            ray_dir = reflected_dir
            throughput *= mirror_reflectivity

    return final_color

@ti.kernel
def render():
    camera_pos = tm.vec3(0.0, 0.0, 5.0)

    for x, y in ti.ndrange(res[0], res[1]):
        fx = (x + 0.5) / res[0]
        fy = (y + 0.5) / res[1]

        u = (fx - 0.5) * 2.0 * (res[0] / res[1])
        v = (fy - 0.5) * 2.0

        ray_dir = tm.normalize(tm.vec3(u, v, -1.0))

        color = trace_ray(camera_pos, ray_dir)

        img[x, y] = color

window = ti.ui.Window("Whitted-Style Ray Tracing", res)
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:
    with gui.sub_window("Light Controls", 0.0, 0.0, 0.35, 0.25):
        gui.text("Light Position")
        light_pos[None].x = gui.slider_float("Light X", light_pos[None].x, -5.0, 5.0)
        light_pos[None].y = gui.slider_float("Light Y", light_pos[None].y, -5.0, 10.0)
        light_pos[None].z = gui.slider_float("Light Z", light_pos[None].z, -5.0, 10.0)

    with gui.sub_window("Ray Tracing Settings", 0.0, 0.27, 0.35, 0.15):
        gui.text("Ray Tracing")
        max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)

    render()
    canvas.set_image(img)
    window.show()