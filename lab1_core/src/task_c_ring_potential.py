import numpy as np


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """计算单个点的电势"""
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    
    r = np.sqrt((x - x_ring)**2 + (y - y_ring)**2 + z**2)
    integrand = 1.0 / r
    integral = np.sum(integrand) * dphi
    
    return (q / (2 * np.pi)) * integral


def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    """在 yz 网格上计算电势矩阵"""
    
    # 处理一维数组输入（测试要求）
    if y_grid.ndim == 1 and z_grid.ndim == 1:
        Y, Z = np.meshgrid(y_grid, z_grid)
    else:
        Y, Z = y_grid, z_grid
    
    # 预计算环上的点
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    
    # 向量化计算
    V = np.zeros_like(Y, dtype=float)
    
    for i in range(n_phi):
        r = np.sqrt((x0 - x_ring[i])**2 + (Y - y_ring[i])**2 + Z**2)
        V += 1.0 / r
    
    V *= (q / (2 * np.pi)) * dphi
    
    return V


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """轴上电势解析解（用于验证）"""
    return q / np.sqrt(a * a + z * z)


def compute_electric_field(V, y_grid, z_grid):
    """计算电场分量 Ey, Ez"""
    dy = y_grid[0, 1] - y_grid[0, 0]
    dz = z_grid[1, 0] - z_grid[0, 0]
    
    Ey = -np.gradient(V, dy, axis=1)
    Ez = -np.gradient(V, dz, axis=0)
    
    return Ey, Ez


def plot_ring_potential(y_range=(-2, 2), z_range=(-2, 2), resolution=100, a=1.0, q=1.0):
    """绘制 yz 平面的等势线和电场线"""
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    Y, Z = np.meshgrid(y, z)
    
    print("Calculating potential...")
    V = ring_potential_grid(y, z, x0=0.0, a=a, q=q)  # 传入一维数组
    
    print("Calculating electric field...")
    Ey, Ez = compute_electric_field(V, Y, Z)
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 等势线
    levels = np.linspace(V.min(), V.max(), 20)
    contour = ax.contour(Y, Z, V, levels=levels, cmap='viridis', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 电场流线
    ax.streamplot(Y, Z, Ey, Ez, color='red', linewidth=0.5, density=1.5)
    
    # 圆环位置
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(a*np.cos(theta), a*np.sin(theta), 'b-', linewidth=2, label='Ring')
    
    ax.set_xlabel('y', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title('Electric Potential and Field of a Charged Ring (x=0 plane)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('ring_potential.png', dpi=150)
    plt.show()
    
    return V, Ey, Ez


if __name__ == "__main__":
    V, Ey, Ez = plot_ring_potential()