# Phong 光照模型实验报告

## 一、实验目标

### 1.1 理论理解
理解并掌握局部光照的基本原理，区分环境光（Ambient）、漫反射（Diffuse）和镜面高光（Specular）。

### 1.2 数学基础
熟练掌握三维空间中的向量运算，包括法向量计算、光线方向、视线方向与反射向量。

### 1.3 工程实践
掌握如何利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数，直观感受各个参数对渲染结果的影响。

## 二、实验原理

### 2.1 Phong 光照模型

Phong 光照模型是一种经典的计算机图形学经验模型，它将物体表面反射的光分为三个独立的计算分量，最终将它们叠加得到像素颜色：

$$I = I_{ambient} + I_{diffuse} + I_{specular}$$

### 2.2 各分量详解

#### 环境光 (Ambient)
模拟场景中经过多次反射后均匀分布的背景光。
$$I_{ambient} = K_a \times C_{light} \times C_{object}$$

#### 漫反射 (Diffuse)
模拟粗糙表面向各个方向均匀散射的光，强度与光线入射角的余弦值成正比（Lambert 定律）。
$$I_{diffuse} = K_d \times \max(0, \mathbf{N} \cdot \mathbf{L}) \times C_{light} \times C_{object}$$

#### 镜面高光 (Specular)
模拟光滑表面反射的强光，强度与观察方向和理想反射方向的夹角有关。
$$I_{specular} = K_s \times \max(0, \mathbf{R} \cdot \mathbf{V})^n \times C_{light}$$

### 2.3 符号说明

| 符号 | 含义 |
|------|------|
| $\mathbf{N}$ | 表面法向量 |
| $\mathbf{L}$ | 指向光源的方向向量 |
| $\mathbf{V}$ | 指向摄像机的方向向量 |
| $\mathbf{R}$ | 光线的理想反射向量 |
| $K_a, K_d, K_s$ | 环境光、漫反射、镜面高光系数 |
| $n$ | 高光指数 (Shininess) |
| $C_{light}$ | 光源颜色 |
| $C_{object}$ | 物体颜色 |

### 2.4 反射向量计算

$$\mathbf{R} = 2 \times (\mathbf{N} \cdot \mathbf{L}) \times \mathbf{N} - \mathbf{L}$$

## 三、实验环境

- **编程语言**: Python 3.12
- **渲染框架**: Taichi 1.7.4
- **窗口尺寸**: 640 × 480

### 3.1 场景配置

| 参数 | 值 |
|------|-----|
| 摄像机位置 | (0, 0, 5) |
| 点光源位置 | (2, 3, 4) |
| 光源颜色 | (1.0, 1.0, 1.0) 白色 |
| 背景颜色 | (0.0, 0.1, 0.1) 深青色 |

### 3.2 几何体配置

#### 红色球体 (Red Sphere)
- 圆心坐标: (-1.2, -0.2, 0)
- 半径: 1.2
- 基础颜色: (0.8, 0.1, 0.1) 深红色

#### 紫色圆锥 (Purple Cone)
- 顶点坐标: (1.2, 1.2, 0)
- 底面高度: y = -1.4
- 底面半径: 1.2
- 基础颜色: (0.6, 0.2, 0.8) 紫色

## 四、代码实现

### 4.1 光线投射 (Ray Casting)

为屏幕上的每一个像素发射一条射线，从摄像机位置 $(0, 0, 5)$ 出发，穿过屏幕平面 $z = 4$ 上的像素点。

```python
ray_origin = tm.vec3(0.0, 0.0, 5.0)
ray_dir = tm.normalize(tm.vec3(u, v, -1.0))
```

### 4.2 光线求交算法

#### 球体求交
使用标准的光线-球体求交公式：

```python
oc = ray_origin - center
a = dot(ray_dir, ray_dir)
b = 2.0 * dot(oc, ray_dir)
c = dot(oc, oc) - radius * radius
discriminant = b * b - 4.0 * a * c
```

#### 圆锥求交
圆锥求交需要解二次方程，并检查交点是否在圆锥的侧面和底面范围内。

### 4.3 Z-buffer 深度测试

核心要求：实现类似 Z-buffer 的深度竞争逻辑。如果射线同时击中两个物体，选择距离摄像机最近的交点（即最小的正数 $t$）进行着色。

```python
min_t = 1e10
if hit_sphere == 1 and t_sphere > 0.01 and t_sphere < min_t:
    min_t = t_sphere
    color = phong_shading(...)

if hit_cone == 1 and t_cone > 0.01 and t_cone < min_t:
    min_t = t_cone
    color = phong_shading(...)
```

### 4.4 Phong 着色器实现

```python
def phong_shading(pos, normal, view_dir, object_color):
    # Ambient
    ambient = Ka[None] * light_color * object_color

    # Diffuse
    L = normalize(light_pos - pos)
    diffuse_factor = max(0.0, dot(normal, L))
    diffuse = Kd[None] * diffuse_factor * light_color * object_color

    # Specular
    R = 2.0 * dot(normal, L) * normal - L
    R = normalize(R)
    V = -view_dir
    spec_factor = max(0.0, dot(R, V))
    specular = Ks[None] * pow(spec_factor, Shininess[None]) * light_color

    return ambient + diffuse + specular
```

### 4.5 UI 交互面板

使用 Taichi 的 `ti.ui.Window` 和 `gui` 模块创建交互窗口，提供 4 个滑动条控件：

| 参数 | 含义 | 范围 | 默认值 |
|------|------|------|--------|
| Ka | 环境光系数 | 0.0 ~ 1.0 | 0.2 |
| Kd | 漫反射系数 | 0.0 ~ 1.0 | 0.7 |
| Ks | 镜面高光系数 | 0.0 ~ 1.0 | 0.5 |
| Shininess | 高光指数 | 1.0 ~ 128.0 | 32.0 |

## 五、参数影响分析

### 5.1 环境光系数 $K_a$
- **作用**: 控制物体在完全没有直接光照时的亮度
- **影响**: 值越大，物体在阴影区域越亮；值越小，阴影区域越暗甚至接近黑色

### 5.2 漫反射系数 $K_d$
- **作用**: 控制漫反射光的强度
- **影响**: 值越大，物体受光面越亮；值越小，物体表面越暗淡

### 5.3 镜面高光系数 $K_s$
- **作用**: 控制镜面反射（高光）的强度
- **影响**: 值越大，高光区域越亮；值越小，高光越不明显

### 5.4 高光指数 $n$ (Shininess)
- **作用**: 控制高光区域的聚散程度
- **影响**:
  - 值越大，高光区域越小越集中，表面越光滑
  - 值越小，高光区域越大越分散，表面越粗糙

## 六、运行方式

```bash
python phong_lighting.py
```

## 七、结论

本次实验成功实现了基于 Phong 光照模型的光线投射渲染系统。通过 Taichi 框架的并行计算能力，在 GPU 上高效完成了：

1. 红色球体和紫色圆锥的光线求交计算
2. Z-buffer 深度测试确保正确的遮挡关系
3. Phong 三分量光照着色
4. 实时可调的 UI 参数控制面板

实验直观地展示了 $K_a$、$K_d$、$K_s$ 和 Shininess 四个参数对渲染效果的影响，加深了对局部光照模型的理解。