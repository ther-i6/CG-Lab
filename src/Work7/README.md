# 质点-弹簧模型物理模拟实验报告

**姓名**：马浩宇
**学号**：202411081004
## 一、实验目标
本实验旨在通过 Taichi 框架实现质点-弹簧模型的物理模拟，深入理解以下内容：

1. **动态场景渲染**：使用 Taichi GGUI 构建 3D 场景，学习交互面板的编写
2. **质点-弹簧模型**：掌握基于物理的弹力与阻尼力计算方法，处理数值爆炸问题
3. **数值积分方法**：实现并比较三种常见数值积分求解器（显式欧拉、半隐式欧拉、隐式欧拉）
4. **GPU 编程基础**：学习 Taichi 中的 `ti.kernel` 与 `ti.func`，理解并行计算中的状态同步

---

## 二、实验原理
### 2.1 质点-弹簧模型 (Mass-Spring Model)
质点-弹簧系统是计算机图形学中经典的变形体模拟方法。将布料离散化为网格状的质点集合，质点之间通过弹簧相连。

**胡克定律（弹力公式）**：

$ f_{a} = -k_{s} (|x_a - x_b| - l) \frac{x_a - x_b}{|x_a - x_b|} $

其中：

+ $ k_s $ 为弹簧的劲度系数
+ $ l $ 为弹簧的原长
+ $ x $ 为质点位置

**阻尼力公式**：

$ f_{d} = -k_{d} v_{a} $

其中 $ k_d $ 为阻尼系数，用于防止系统能量无限增加导致发散。

### 2.2 数值积分方法
根据牛顿第二定律 $ F = ma $，在离散时间步 $ \Delta t $ 内通过数值积分更新质点状态。

#### 显式欧拉 (Explicit Euler)
$ x_{t+1} = x_{t} + v_{t} \Delta t $

$ v_{t+1} = v_{t} + a_{t} \Delta t $

**特点**：计算简单，但稳定性较差，时间步长较小时可能出现数值爆炸。

#### 半隐式欧拉 (Semi-Implicit / Symplectic Euler)
$ v_{t+1} = v_{t} + a_{t} \Delta t $

$ x_{t+1} = x_{t} + v_{t+1} \Delta t $

**特点**：先更新速度，再用更新后的速度更新位置。比显式欧拉更稳定，能较好地保持系统能量。

#### 隐式欧拉 (Implicit / Backward Euler)
$ v_{t+1} = v_{t} + a_{t+1} \Delta t $

$ x_{t+1} = x_{t} + v_{t+1} \Delta t $

**特点**：使用未来时刻的状态计算受力，稳定性最好，但计算复杂度较高（本实验使用定点迭代法近似求解）。

---

## 三、实现步骤
### 3.1 场景初始化
定义布料网格大小（8×8），初始化质点位置、速度和弹簧拓扑结构：

```python
N = 8                          # 网格分辨率
GRID_SIZE = 0.3                # 网格间距
TOTAL_POINTS = N * N

pos = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_POINTS)
vel = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_POINTS)
```

### 3.2 力学计算与防爆处理
**受力计算**：重力 + 阻尼力 + 弹簧力

```python
force = mass_value * gravity - kd * vel[i]

# 水平邻居弹簧力
if i % N != 0:
    j_idx = i - 1
    diff = pos[i] - pos[j_idx]
    dist = diff.norm()
    force += -ks_structure * (dist - GRID_SIZE) * (diff / dist)
```

**速度钳制**：限制最大速度防止数值爆炸

```python
speed = vel[i].norm()
if speed > max_velocity:
    vel[i] = vel[i] * (max_velocity / speed)
```

### 3.3 积分求解器实现
三种积分方法均实现为 `@ti.kernel`，将受力计算和位置/速度更新合并在同一个 Kernel 中，最小化 Kernel 启动开销。

### 3.4 渲染与 GGUI 交互
使用 Taichi GGUI 构建交互面板：

+ 三种积分方法切换按钮
+ 暂停/恢复功能
+ 重置功能

---

## 四、代码结构
```plain
main.py
├── 参数配置（第5-13行）
│   ├── N: 网格分辨率
│   ├── GRID_SIZE: 网格间距
│   ├── dt: 时间步长
│   ├── ks_structure: 弹簧劲度系数
│   ├── kd: 阻尼系数
│   └── max_velocity: 最大速度限制
│
├── 数据结构定义（第15-26行）
│   ├── pos: 质点位置
│   ├── vel: 质点速度
│   ├── line_verts: 线条数据（GPU存储）
│   ├── paused: 暂停状态
│   └── method: 当前积分方法
│
├── 初始化函数（第28-79行）
│   ├── init_positions(): 初始化质点位置
│   ├── init_lines(): 初始化线条
│   └── update_lines(): 更新线条（GPU内完成）
│
├── 数值积分求解器（第81-210行）
│   ├── step_explicit(): 显式欧拉
│   ├── step_semi_implicit(): 半隐式欧拉
│   └── step_implicit_iter(): 隐式欧拉
│
└── 主函数（第218-279行）
    ├── GUI控制面板
    ├── 物理更新循环
    └── 渲染输出
```

---

## 五、实验结果分析
### 5.1 稳定性对比
| 积分方法 | 稳定性 | 特点 |
| --- | --- | --- |
| 显式欧拉 | 较差 | 时间步长较大时易发散，但计算最快 |
| 半隐式欧拉 | 较好 | 能保持系统能量，是平衡稳定性和效率的最佳选择 |
| 隐式欧拉 | 最好 | 无条件稳定，但计算复杂度较高 |


### 5.2 性能优化策略
1. **GPU加速渲染**：线条数据存储在 GPU 上，`update_lines()` 在 GPU 内完成，消除 CPU-GPU 数据传输瓶颈
2. **合并 Kernel**：将受力计算和位置更新合并在同一 Kernel 中，减少 Kernel 启动次数
3. **简化边界检查**：使用简单条件判断替代复杂逻辑

### 5.3 参数影响分析
+ **劲度系数 (**$ k_s $**)**：值越大，弹簧越硬，布料越不易变形
+ **阻尼系数 (**$ k_d $**)**：值越大，能量衰减越快，摆动幅度越小
+ **时间步长 (**$ \Delta t $**)**：值越大，每帧计算量越小，但稳定性变差

---

## 六、运行说明
### 6.1 环境要求
+ Python 3.10+
+ Taichi 1.7.0+
+ CUDA 支持

### 6.2 运行命令
```bash
cd Work7
python main.py
```

### 6.3 交互说明
| 按钮 | 功能 |
| --- | --- |
| Explicit | 切换到显式欧拉方法 |
| Semi-Impl | 切换到半隐式欧拉方法 |
| Implicit | 切换到隐式欧拉方法 |
| Pause | 暂停/恢复模拟 |
| Reset | 重置布料到初始状态 |




### 6.4 演示视频
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/gif/66117676/1780396370662-68f32ba0-5803-44e0-a6d0-02592b3ac27e.gif)

---

## 七、实验结论
通过本次实验，深入理解了质点-弹簧模型的物理模拟原理，掌握了三种数值积分方法的实现和对比分析。实验结果表明：

1. **半隐式欧拉**是平衡稳定性和计算效率的最佳选择，在大多数场景下推荐使用
2. **GPU 并行计算**能显著提升物理模拟的性能
3. **速度钳制**是防止数值爆炸的有效手段
4. **减少 Kernel 启动次数**和**避免 CPU-GPU 数据传输**是优化 Taichi 程序性能的关键

---

## 八、参考文献
1. Bridson, R. (2008). _Fluid Simulation for Computer Graphics_. CRC Press.
2. Taichi Documentation. [https://docs.taichi-lang.org/](https://docs.taichi-lang.org/)
3. Baraff, D., & Witkin, A. (1998). Large Steps in Cloth Simulation. _SIGGRAPH_.

