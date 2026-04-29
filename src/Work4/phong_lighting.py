import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu, offline_cache=False)

res = (640, 480)

Ka = ti.field(dtype=float, shape=())
Kd = ti.field(dtype=float, shape=())
Ks = ti.field(dtype=float, shape=())
Shininess = ti.field(dtype=float, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
Shininess[None] = 32.0

sphere_center = tm.vec3(-1.2, -0.2, 0.0)
sphere_radius = 1.2
sphere_color = tm.vec3(0.8, 0.1, 0.1)

cone_apex = tm.vec3(1.2, 1.2, 0.0)
cone_height = 2.6
cone_radius = 1.2
cone_color = tm.vec3(0.6, 0.2, 0.8)

light_pos = tm.vec3(2.0, 3.0, 4.0)
light_color = tm.vec3(1.0, 1.0, 1.0)
bg_color = tm.vec3(0.0, 0.1, 0.1)

img = ti.Vector.field(3, dtype=float, shape=res)

@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    t = -1.0
    hit_pos = tm.vec3(0.0)
    normal = tm.vec3(0.0, 1.0, 0.0)
    hit = 0

    oc = ray_origin - center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    t = -1.0
    if discriminant > 0:
        t0 = (-b - tm.sqrt(discriminant)) / (2.0 * a)
        t1 = (-b + tm.sqrt(discriminant)) / (2.0 * a)
        if t0 > 0.01:
            t = t0
        else:
            if t1 > 0.01:
                t = t1

    if t > 0.01:
        hit_pos = ray_origin + t * ray_dir
        normal = tm.normalize(hit_pos - center)
        hit = 1

    return hit, t, hit_pos, normal

@ti.func
def intersect_cone(ray_origin, ray_dir, apex, height, radius):
    t = -1.0
    hit_pos = tm.vec3(0.0)
    normal = tm.vec3(0.0, 1.0, 0.0)
    hit = 0

    k = radius / height
    k2 = k * k

    apex_to_origin = ray_origin - apex
    dir_dot_ax = ray_dir.y
    orig_dot_ax = apex_to_origin.y

    a = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z - k2 * dir_dot_ax * dir_dot_ax
    b = 2.0 * (ray_dir.x * apex_to_origin.x + ray_dir.z * apex_to_origin.z - k2 * dir_dot_ax * orig_dot_ax)
    c = apex_to_origin.x * apex_to_origin.x + apex_to_origin.z * apex_to_origin.z - k2 * orig_dot_ax * orig_dot_ax

    disc = b * b - 4.0 * a * c
    t = -1.0

    if disc > 0:
        t0 = (-b - tm.sqrt(disc)) / (2.0 * a)
        t1 = (-b + tm.sqrt(disc)) / (2.0 * a)

        base_y = apex.y - height
        found = 0
        best_t = 1e10

        if t0 > 0.01:
            pos = ray_origin + t0 * ray_dir
            if pos.y > base_y and pos.y < apex.y and t0 < best_t:
                best_t = t0
                hit_pos = pos
                normal = tm.normalize(tm.vec3(pos.x - apex.x, k2 * (apex.y - pos.y), pos.z - apex.z))
                found = 1

        if t1 > 0.01:
            pos = ray_origin + t1 * ray_dir
            if pos.y > base_y and pos.y < apex.y and t1 < best_t:
                best_t = t1
                hit_pos = pos
                normal = tm.normalize(tm.vec3(pos.x - apex.x, k2 * (apex.y - pos.y), pos.z - apex.z))
                found = 1

        if found == 1:
            t = best_t
            hit = 1

        if found == 0:
            t_base = (apex.y - height - ray_origin.y) / ray_dir.y if abs(ray_dir.y) > 0.001 else -1.0
            if t_base > 0.01:
                base_pos = ray_origin + t_base * ray_dir
                dist_sq = (base_pos.x - apex.x) * (base_pos.x - apex.x) + (base_pos.z - apex.z) * (base_pos.z - apex.z)
                if dist_sq < radius * radius:
                    t = t_base
                    hit_pos = base_pos
                    normal = tm.vec3(0.0, -1.0, 0.0)
                    hit = 1

    return hit, t, hit_pos, normal

@ti.func
def phong_shading(pos, normal, view_dir, object_color):
    ambient = Ka[None] * light_color * object_color

    L = tm.normalize(light_pos - pos)
    diffuse_factor = tm.max(0.0, tm.dot(normal, L))
    diffuse = Kd[None] * diffuse_factor * light_color * object_color

    R = 2.0 * tm.dot(normal, L) * normal - L
    R = tm.normalize(R)
    V = -view_dir
    spec_factor = tm.max(0.0, tm.dot(R, V))
    specular = Ks[None] * tm.pow(spec_factor, Shininess[None]) * light_color

    return ambient + diffuse + specular

@ti.kernel
def render():
    for x, y in ti.ndrange(res[0], res[1]):
        fx = (x + 0.5) / res[0]
        fy = (y + 0.5) / res[1]
        
        u = (fx - 0.5) * 2.0 * (res[0] / res[1])
        v = (fy - 0.5) * 2.0

        ray_dir = tm.normalize(tm.vec3(u, v, -1.0))
        ray_origin = tm.vec3(0.0, 0.0, 5.0)

        hit_sphere, t_sphere, pos_sphere, norm_sphere = intersect_sphere(
            ray_origin, ray_dir, sphere_center, sphere_radius)
        hit_cone, t_cone, pos_cone, norm_cone = intersect_cone(
            ray_origin, ray_dir, cone_apex, cone_height, cone_radius)

        color = bg_color
        min_t = 1e10

        if hit_sphere == 1 and t_sphere > 0.01 and t_sphere < min_t:
            min_t = t_sphere
            color = phong_shading(pos_sphere, norm_sphere, ray_dir, sphere_color)

        if hit_cone == 1 and t_cone > 0.01 and t_cone < min_t:
            min_t = t_cone
            color = phong_shading(pos_cone, norm_cone, ray_dir, cone_color)

        img[x, y] = color

window = ti.ui.Window("Phong Lighting Model", res)
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:
    with gui.sub_window("Phong Lighting Controls", 0.0, 0.0, 0.35, 0.35):
        gui.text("Material Parameters")
        Ka[None] = gui.slider_float("Ka (Ambient)", Ka[None], 0.0, 1.0)
        Kd[None] = gui.slider_float("Kd (Diffuse)", Kd[None], 0.0, 1.0)
        Ks[None] = gui.slider_float("Ks (Specular)", Ks[None], 0.0, 1.0)
        Shininess[None] = gui.slider_float("Shininess", Shininess[None], 1.0, 128.0)

    render()
    canvas.set_image(img)
    window.show()