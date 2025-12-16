"""
运动学计算函数

计算 dimuon 系统的运动学量，如不变质量、deltaR 等。
"""

import numpy as np


# 物理常数
MUON_MASS = 0.1056583755  # GeV


def assemble_m_inv(
    a_M: np.ndarray,
    a_pt: np.ndarray,
    a_eta: np.ndarray,
    a_phi: np.ndarray,
    b_M: np.ndarray,
    b_pt: np.ndarray,
    b_eta: np.ndarray,
    b_phi: np.ndarray,
) -> tuple:
    """
    计算两个粒子组成的母粒子的不变质量和其他运动学量
    
    物理假设：
    - 使用相对论运动学
    - 假设两个粒子是 muon（使用 muon_mass）
    
    Args:
        a_M, a_pt, a_eta, a_phi: 第一个粒子的质量、横动量、赝快度、方位角
        b_M, b_pt, b_eta, b_phi: 第二个粒子的质量、横动量、赝快度、方位角
        
    Returns:
        (mother_M, mother_pt, mother_eta, mother_phi, good_event_indices)
        - mother_M: 母粒子不变质量
        - mother_pt: 母粒子横动量
        - mother_eta: 母粒子赝快度
        - mother_phi: 母粒子方位角
        - good_event_indices: 有效事件索引（M^2 >= 0 且 pt > 0）
    """
    # 计算能量
    a_E = np.sqrt(a_M**2 + (a_pt * np.cosh(a_eta)) ** 2)
    b_E = np.sqrt(b_M**2 + (b_pt * np.cosh(b_eta)) ** 2)
    
    # 计算动量分量
    a_px = a_pt * np.cos(a_phi)
    b_px = b_pt * np.cos(b_phi)
    
    a_py = a_pt * np.sin(a_phi)
    b_py = b_pt * np.sin(b_phi)
    
    a_pz = a_pt * np.sinh(a_eta)
    b_pz = b_pt * np.sinh(b_eta)
    
    # 母粒子四动量
    mother_E = a_E + b_E
    mother_px = a_px + b_px
    mother_py = a_py + b_py
    mother_pz = a_pz + b_pz
    
    # 不变质量平方
    M_sq_cands = mother_E**2 - mother_px**2 - mother_py**2 - mother_pz**2
    
    # 有效事件筛选（M^2 >= 0 且 pt > 0）
    good_event_indices = (M_sq_cands >= 0) & (
        np.sqrt(mother_px**2 + mother_py**2) > 0
    )
    
    # 计算不变质量（只对有效事件）
    mother_M = np.sqrt(np.maximum(M_sq_cands, 0))
    mother_pt = np.sqrt(mother_px**2 + mother_py**2)
    mother_eta = np.arcsinh(mother_pz / np.maximum(mother_pt, 1e-10))
    mother_phi = np.arctan2(mother_py, mother_px)
    
    return mother_M, mother_pt, mother_eta, mother_phi, good_event_indices


def calculate_deltaR(
    phi_0: np.ndarray, phi_1: np.ndarray, eta_0: np.ndarray, eta_1: np.ndarray
) -> np.ndarray:
    """
    计算两个粒子之间的 deltaR
    
    deltaR = sqrt(delta_phi^2 + delta_eta^2)
    
    Args:
        phi_0, phi_1: 两个粒子的方位角
        eta_0, eta_1: 两个粒子的赝快度
        
    Returns:
        deltaR 数组
    """
    delta_phi = np.abs(phi_0 - phi_1)
    # 调整到 (-pi, pi) 范围
    delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)
    delta_R = np.sqrt(delta_phi**2 + (eta_0 - eta_1) ** 2)
    
    return delta_R


def calculate_deltaPT(
    pt_0: np.ndarray, pt_1: np.ndarray, phi_0: np.ndarray, phi_1: np.ndarray
) -> np.ndarray:
    """
    计算两个粒子横动量之间的差值
    
    deltaPT = |pt_0 - pt_1|
    
    Args:
        pt_0, pt_1: 两个粒子的横动量
        phi_0, phi_1: 两个粒子的方位角（未使用，保留接口一致性）
        
    Returns:
        deltaPT 数组
    """
    return np.abs(pt_0 - pt_1)


# 导出 muon_mass 作为模块级常量
muon_mass = MUON_MASS

