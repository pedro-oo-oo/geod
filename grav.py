# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:34:18 2025

@author: pnapi

Gravimetry fraction of geod package

"""

import numpy as np

from . import const
from . import geod
from . import funcs
from . import obj

def dg_height(height):
    """
    Returns dg_h correction to gravimetry calculations.
    Unit: mGal
    """
    return round(0.3086 * height, 4)

def drift():
    ...
    
def gamma0_grs80(phi):
    gamma_0 = (978032.66 
               * (1 + 0.0053024*phi.sin()**2 - 0.0000058*phi.sin(2)**2))
    return (gamma_0)

def dg_bouguer(height, density=2.67):
    """
    Returns dg_bouguer correction to gravimetry calculations.
    Unit: mGal
    """
    return round(-0.04192 * density * height, 4)

def dg_poincaryprey(height, density=2.67):
    return round((0.3086 - 2*0.04192 * density) * height, 4)

def anomaly(phi, g_measured, dg):
    g0 = gamma0_grs80(phi)
    delta = g_measured + dg - gamma0_grs80(phi)
    return round(delta, 4)
    
def height_dif_geopot(h_difs, g_args):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    try:    
        return sum(((g_args[i]-10**6) / (10**6) * h_difs[i]) for i in range(len(h_difs)))
    except TypeError :
        return (g_args - 10**6) / (10**6) * h_difs
    
def height_dif_dynamic(h_difs, g_args):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    g45 = gamma0_grs80(obj.Angle(np.pi/4))
    try:    
        return sum(((g_args[i]-g45) / (g45) * h_difs[i]) for i in range(len(h_difs)))
    except TypeError :
        return (g_args - g45) / (g45) * h_difs
    
def height_dif_orthometric(h_a, h_b, h_difs, g_args):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    g45 = gamma0_grs80(obj.Angle(np.pi/4))
    ga = g_args[ 0] + dg_poincaryprey(h_a/2) 
    ga = (ga - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    gb = g_args[-1] + dg_poincaryprey(h_b/2) 
    gb = (gb - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    try:    
        return sum(((g_args[i]-g45) / (g45) * h_difs[i]) for i in range(len(h_difs))) + ga - gb
    except TypeError :
        return (g_args[0] - g45) / (g45) * h_difs + ga - gb 
    
def height_dif_normal(phi_a, h_a, phi_b, h_b, h_difs, g_args):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    g45 = gamma0_grs80(obj.Angle(np.pi/4))
    ga = gamma0_grs80(phi_a) - dg_height(h_a/2) 
    ga = (ga - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    gb = gamma0_grs80(phi_b) - dg_height(h_b/2)
    gb = (gb - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    try:    
        return sum(((g_args[i]-g45) / (g45) * h_difs[i]) for i in range(len(h_difs))) + ga - gb
    except TypeError:
        return (g_args - g45) / (g45) * h_difs + ga - gb 
    
def height_geopot(Cp):
    return Cp / 10**6

def height_dynamic(Cp):
    return Cp / gamma0_grs80(obj.Angle(np.pi/4))

def height_orthometric(Cp, g_measured, h):
    g = g_measured + dg_poincaryprey(h/2) 
    return Cp / g

def height_normal(Cp, phi, h):
    g = gamma0_grs80(phi) - dg_height(h/2) 
    return Cp / g