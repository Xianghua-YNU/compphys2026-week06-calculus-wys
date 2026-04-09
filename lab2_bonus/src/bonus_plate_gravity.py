import numpy as np


G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    # TODO D1: 使用二维高斯-勒让德积分实现双重积分
    raise NotImplementedError("TODO D1")


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    # TODO D2: 计算方板中心正上方 z 位置的 Fz
    raise NotImplementedError("TODO D2")


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    # TODO D3: 返回 z_values 对应的 Fz 数组
    raise NotImplementedError("TODO D3")
