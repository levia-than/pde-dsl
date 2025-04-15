# pde-dsl

A DSL for Solving PDE.

# 测试集目标方程

## 热传导方程

$$
\frac{\partial{u}}{\partial{t}} = \Delta u
$$

## 浅水方程（SWE）

### 一维浅水方程

$$
\left\{
\begin{array}
{ll}\frac{\partial}{\partial t}h+\frac{\partial}{\partial x}hv=0 & \quad(1) \\
\frac{\partial}{\partial t}hv+\frac{\partial}{\partial x}\left(hv^2+\frac{gh^2}{2}\right)=0 & \quad(2)
\end{array}\right.
$$

方程中，$h(x,t)$代表水在时间t位置x的深度，$v(x,t)$代表水在位置的流速。

### 二维潜水方程

### 二维圣维南方程

目标方程：

$$
\begin{aligned}
& \frac{\partial h}{\partial t}+\frac{\partial(h u)}{\partial x}+\frac{\partial(h v)}{\partial y}=q_{\mathrm{L}} \\
& \frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}+g \frac{\partial h}{\partial x}=b_x^{\prime} \\
& \frac{\partial v}{\partial t}+u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}+g \frac{\partial h}{\partial y}=b_y^{\prime}
\end{aligned}
$$

其中，

$b_x^{\prime}=-\frac{1}{\rho} \frac{\partial p_{\mathrm{a}}}{\partial x}-g \frac{\partial z_{\mathrm{b}}}{\partial x}+\frac{\tau_{\mathrm{as}}-\tau_{\mathrm{br}}}{h}+F_{\mathrm{b} x}$

### 最简情况

### 一般unregular网格形

# 求解方法

## fdm

## fvm
