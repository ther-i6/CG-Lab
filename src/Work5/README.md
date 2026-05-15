# Whitted-Style 光线追踪实验报告
## 实验概述
本实验实现了经典的 Whitted-Style 光线追踪算法，通过发射次级射线（Secondary Rays）来实现硬阴影（Hard Shadows）和理想镜面反射（Perfect Reflection）。

## 实验目标
### 理论理解
+ 理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别
+ 光线投射：仅追踪主光线与物体的交点，计算直接光照
+ 光线追踪：追踪光线在场景中的多次弹射，包括反射、折射等

### 全局光照
+ 掌握如何通过发射次级射线实现硬阴影
+ 掌握理想镜面反射的实现方法

### GPU 编程思维
+ 学习如何将递归光线追踪算法改写为适合 GPU 并行计算的迭代模式

## 实验原理
### 算法模型
采用经典的 Whitted-Style 光线追踪模型：

1. **主光线（Primary Ray）**：从摄像机出发，穿过像素，进入场景
2. **交点计算**：检测主光线与场景中物体的交点
3. **阴影测试**：从交点向光源方向发射暗影射线
4. **材质分支**：根据材质类型决定光线后续行为

### 反射向量公式
$ \mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in} \cdot \mathbf{N})\mathbf{N} $

其中：

+ $ \mathbf{L}_{in} $ 为入射光线方向
+ $ \mathbf{N} $ 为表面法向量

### 迭代光线追踪流程
```plain
初始化 throughput = 1.0, final_color = 0
for bounce in range(max_bounces):
    追踪光线与场景交点
    if 未击中任何物体:
        break
    if 击中漫反射物体:
        计算光照颜色
        final_color += throughput * color
        break
    if 击中镜面物体:
        计算反射方向
        更新光线起点和方向
        throughput *= 反射率
return final_color
```

## 场景设置
### 几何定义
| 物体 | 类型 | 位置 | 半径 | 材质 | 颜色 |
| --- | --- | --- | --- | --- | --- |
| 地面 | 无限平面 | y = -1.0 | - | 漫反射 | 棋盘格纹理 |
| 红球 | 球体 | (-1.5, 0.0, 0) | 1.0 | 漫反射 | 红色 |
| 银球 | 球体 | (1.5, 0.0, 0) | 1.0 | 镜面 | 银色 |


### 棋盘格纹理实现
通过交点的 x 和 z 坐标奇偶性判断：

```python
checker = int(floor(hit_pos.x) + floor(hit_pos.z)) % 2
if checker == 0:
    color = 白色
else:
    color = 黑色
```

## 关键技术实现
### 1. 物体相交检测
#### 球体相交
```python
oc = ray_origin - center
a = dot(ray_dir, ray_dir)
b = 2.0 * dot(oc, ray_dir)
c = dot(oc, oc) - radius * radius
discriminant = b * b - 4.0 * a * c
```

#### 平面相交
```python
if abs(ray_dir.y) > epsilon:
    t = (ground_y - ray_origin.y) / ray_dir.y
```

### 2. 硬阴影实现
```python
shadow_ray_origin = hit_pos + normal * epsilon
shadow_ray_dir = normalize(light_pos - shadow_ray_origin)
```

### 3. 解决自相交 Bug（Shadow Acne）
**核心避坑点**：将反射射线和暗影射线的起点沿法线方向向外偏移一个极小值：

$ \mathbf{P}_{new} = \mathbf{P} + \mathbf{N} \times \epsilon $

其中 $ \epsilon = 1e-4 $

### 4. 材质系统
| 材质 ID | 类型 | 行为 |
| --- | --- | --- |
| 0 | 漫反射 | 计算光照后终止光线传播 |
| 1 | 镜面反射 | 计算反射方向，继续传播 |
| 2 | 地面 | 同漫反射，带棋盘格纹理 |


## UI 交互面板
提供以下滑动条控件：

| 控件 | 范围 | 默认值 | 功能 |
| --- | --- | --- | --- |
| Light X | -5.0 ~ 5.0 | 2.0 | 光源 X 坐标 |
| Light Y | -5.0 ~ 10.0 | 3.0 | 光源 Y 坐标 |
| Light Z | -5.0 ~ 10.0 | 4.0 | 光源 Z 坐标 |
| Max Bounces | 1 ~ 5 | 3 | 最大光线弹射次数 |


## 实验结果分析
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/gif/66117676/1778821887765-30047116-2a44-4165-8d91-9458bbd491b7.gif)

### 不同弹射次数的效果
| 弹射次数 | 效果描述 |
| --- | --- |
| 1 | 无反射，只能看到直接光照 |
| 2 | 一次镜面反射，可以看到镜中世界 |
| 3+ | 多次反射，镜中镜效果 |


### 光照位置对阴影的影响
+ 光源位置改变会实时影响阴影的形状和位置
+ 阴影边缘清晰（硬阴影特性）
+ 当物体位于光源和被照射物体之间时产生阴影

## 代码结构
```plain
main.py
├── 初始化配置
│   ├── 分辨率设置 (640x480)
│   ├── 光源位置和颜色
│   └── 材质参数
├── 场景定义
│   ├── 红球参数
│   ├── 银球参数
│   └── 地面参数
├── 相交检测函数
│   ├── intersect_sphere()
│   └── intersect_ground()
├── 光照计算函数
│   ├── check_shadow()
│   └── compute_diffuse_color()
├── 纹理函数
│   └── get_checkerboard_color()
├── 光线追踪核心
│   └── trace_ray() - 迭代光线弹射
├── 渲染内核
│   └── render()
└── UI 交互循环
    └── 滑动条控件更新
```



## 实验结论
本实验成功实现了 Whitted-Style 光线追踪算法，包括：

1. ✅ 平面与球体的隐式定义
2. ✅ 棋盘格纹理实现
3. ✅ 漫反射与镜面反射材质系统
4. ✅ 硬阴影生成
5. ✅ 迭代式光线弹射（GPU友好）
6. ✅ 自相交 Bug 修复
7. ✅ 交互式 UI 控制面板

通过调节光照位置和最大弹射次数，可以直观观察光线追踪的全局光照效果。

