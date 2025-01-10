# pde-dsl
A DSL for Solving PDE

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
