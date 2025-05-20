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
    
def height_dif_geopot(h_dif, g_a, g_b):
    gi = np.average([g_a, g_b])
    return (gi - 1e6)/1e6 * h_dif
    
def height_dif_dynamic(h_dif, g_a, g_b):
    gi = np.average([g_a, g_b])
    g45 = gamma0_grs80(obj.Angle([45]))
    return (gi - g45)/g45 * h_dif
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    
    
def height_dif_orthometric(h_a, h_b, dhi, g_a, g_b):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    g045 = gamma0_grs80(obj.Angle(np.pi/4))
    gr_a = g_a + dg_height(h_a/2) + 2*dg_bouguer(h_a/2)
    gr_b = g_b + dg_height(h_b/2) + 2*dg_bouguer(h_b/2)
    gi = np.average([g_a, g_b])
    return ((gi-g045) / g045 * dhi
            + (gr_a - g045) / g045 * h_a
            - (gr_b - g045) / g045 * h_b)
    
    
    # ga = g[0] + dg_poincaryprey(h_a/2) 
    # ga = (ga - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    # gb = g_args[-1] + dg_poincaryprey(h_b/2) 
    # gb = (gb - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    
    # try:    
    #     return sum(((g_args[i]-g45) / (g45) * h_difs[i]) for i in range(len(h_difs))) + ga - gb
    # except TypeError :
    #     return (g_args[0] - g45) / (g45) * h_difs + ga - gb 
    
def height_dif_normal(phi_a, h_a, phi_b, h_b, dhi, g_a, g_b):
    # # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    # g45 = gamma0_grs80(obj.Angle(np.pi/4))
    # ga = gamma0_grs80(phi_a) - dg_height(h_a/2) 
    # ga = (ga - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    # gb = gamma0_grs80(phi_b) - dg_height(h_b/2)
    # gb = (gb - gamma0_grs80(obj.Angle(np.pi/4))) / gamma0_grs80(obj.Angle(np.pi/4))
    # try:    
    #     return sum(((g_args[i]-g45) / (g45) * h_difs[i]) for i in range(len(h_difs))) + ga - gb
    # except TypeError:
    #     return (g_args - g45) / (g45) * h_difs + ga - gb 
    
    gi = np.average([g_a, g_b])
    g045 = gamma0_grs80(obj.Angle([np.pi/4]))
    gam_a = gamma0_grs80(phi_a) - 0.3086*(h_a/2)
    gam_b = gamma0_grs80(phi_b) - 0.3086*(h_b/2)
    return ((gi-g045) / g045 * dhi
            + (gam_a - g045) / g045 * h_a
            - (gam_b - g045) / g045 * h_b)
    
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