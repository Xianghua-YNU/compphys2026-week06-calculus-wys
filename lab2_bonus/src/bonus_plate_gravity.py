import numpy as np

G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """二维高斯-勒让德积分"""
    
    # 获取节点和权重
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # x 方向变换参数
    x_mid = (ax + bx) / 2
    x_half = (bx - ax) / 2
    
    # y 方向变换参数
    y_mid = (ay + by) / 2
    y_half = (by - ay) / 2
    
    integral = 0.0
    for i in range(n):
        x = x_mid + x_half * nodes[i]
        wx = weights[i] * x_half
        
        for j in range(n):
            y = y_mid + y_half * nodes[j]
            wy = weights[j] * y_half
            
            integral += wx * wy * func(x, y)
    
    return integral


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    """计算方板中心正上方 z 位置的 Fz"""
    
    # 面密度
    sigma = M_plate / (L * L)
    
    # 被积函数
    def integrand(x, y):
        r2 = x*x + y*y + z*z
        return 1.0 / (r2 ** 1.5)
    
    # 积分区域
    half_L = L / 2
    
    # 二维积分
    integral = gauss_legendre_2d(integrand, -half_L, half_L, -half_L, half_L, n)
    
    # 引力
    Fz = G * sigma * m_particle * z * integral
    
    return Fz


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    """返回 z_values 对应的 Fz 数组"""
    
    Fz_values = []
    for z in z_values:
        Fz = plate_force_z(z, L, M_plate, m_particle, n)
        Fz_values.append(Fz)
    
    return np.array(Fz_values)


# 可选：可视化代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 生成 z 值范围 [0.2, 10]
    z_values = np.linspace(0.2, 10.0, 50)
    
    print("Calculating force curve...")
    Fz_values = force_curve(z_values, n=40)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, Fz_values, 'b-', linewidth=2)
    plt.xlabel('z (m)', fontsize=12)
    plt.ylabel('Fz (N)', fontsize=12)
    plt.title('Gravitational Force vs Height above Square Plate Center', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加物理标注
    plt.text(0.5, max(Fz_values)*0.8, 
             f'Plate: {10}m×{10}m, M={1e4}kg\nParticle mass: 1kg',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('plate_gravity_force.png', dpi=150)
    plt.show()
    
    # 打印几个关键点
    print("\n关键点引力值:")
    for z in [0.2, 1.0, 5.0, 10.0]:
        idx = np.argmin(np.abs(z_values - z))
        print(f"  z = {z:.1f} m, Fz = {Fz_values[idx]:.2e} N")