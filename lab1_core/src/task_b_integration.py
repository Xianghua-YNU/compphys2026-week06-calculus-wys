import math


def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    # TODO B1: 实现复合梯形积分
    raise NotImplementedError("TODO B1")


def simpson_composite(f, a: float, b: float, n: int) -> float:
    # TODO B2: 实现复合 Simpson 积分，并检查 n 为偶数
    raise NotImplementedError("TODO B2")


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # TODO B3: 计算 Debye 积分 I(theta_d/T)
    raise NotImplementedError("TODO B3")
