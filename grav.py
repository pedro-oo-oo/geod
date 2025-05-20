# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:34:18 2025

@author: pnapi

Gravimetry fraction of geod package

"""

import numpy as np

from . import const
from . import geod as geodcore
from . import funcs
from . import obj as geod

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
    # g0 = gamma0_grs80(phi)
    delta = g_measured + dg - gamma0_grs80(phi)
    return round(delta, 4)
    
def height_dif_geopot(h_dif, g_a, g_b):
    gi = np.average([g_a, g_b])
    return (gi - 1e6)/1e6 * h_dif
    
def height_dif_dynamic(h_dif, g_a, g_b):
    gi = np.average([g_a, g_b])
    g45 = gamma0_grs80(geod.Angle([45]))
    return (gi - g45)/g45 * h_dif
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    
    
def height_dif_orthometric(h_a, h_b, dhi, g_a, g_b):
    # assert len(h_difs) == len(g_args), 'h_difs must be of 1 size smaller than g_args'
    g045 = gamma0_grs80(geod.Angle(np.pi/4))
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
    g045 = gamma0_grs80(geod.Angle([np.pi/4]))
    gam_a = gamma0_grs80(phi_a) - 0.3086*(h_a/2)
    gam_b = gamma0_grs80(phi_b) - 0.3086*(h_b/2)
    return ((gi-g045) / g045 * dhi
            + (gam_a - g045) / g045 * h_a
            - (gam_b - g045) / g045 * h_b)
    
def height_geopot(Cp):
    return Cp / 10**6

def height_dynamic(Cp):
    return Cp / gamma0_grs80(geod.Angle(np.pi/4))

def height_orthometric(Cp, g_measured, h):
    g = g_measured + dg_poincaryprey(h/2) 
    return Cp / g

def height_normal(Cp, phi, h):
    g = gamma0_grs80(phi) - dg_height(h/2) 
    return Cp / g


class Height:
    """
    Calculate a point's height in a different system 
    without dealing with all pesky stuff.

    Args:
        C (float | geod.grav.GeopotNum): 
            Geopotential number.
            In gravimetry module we assume the use of CGS units,
            therefore C is given in Gal*m (i think)
        output_type (str): 
            Select the height system you want the object to be printed in.
            'normal / orthometric / dynamic / geopotential'   
            
        These next ones are not mandatory, used only for certain systems,
        you'll be warned if you miss them dw pookie
        
        phi (geod.Angle): 
            Latitude of the point where Height is calculated            
        g (float): 
            Gravitational acceleration, in units of mGal.
        h_appr (float): 
            Approximate height of the point, 
            needed for gravity stuff calculations, i don even know bruh
    """
    def __init__(self, 
                 C: float, 
                 output_type: str | None,
                 phi: geod.Angle | None = None, 
                 g: float | None = None,
                 h_appr: float | None = None) -> None:
        if hasattr(C, 'C'):
            self.C = C.C 
        else:
            self.C = C
        
        self.phi = phi
        self.h_appr = h_appr
        self.g = g
        
        self.output_type = output_type
        
    def convert(self, output_type: str) -> float:
        
        match output_type:
            case 'orthometric':
                assert self.g is not None, 'Unspecified g in Height initialization'
                assert self.h_appr is not None, 'Unspecified h_appr in Height initialization'
                return height_orthometric(self.C, self.g, self.h_appr)
            
            case 'dynamic':
                return height_dynamic(self.C)
            
            case 'normal':
                assert self.phi is not None, 'Unspecified phi in Height initialization'
                assert self.h_appr is not None, 'Unspecified h_appr in Height initialization'
                return height_normal(self.C, self.phi, self.h_appr)
                
            case 'geopotential':
                return height_geopot(self.C)
            
            case _:
                raise TypeError('Invalid output_type, read documentation dude')
                
    def __str__(self) -> str:
        return f'{self.convert(self.output_type):.3f}'
    

class GeopotNum:
    def __init__(self, height: float,
                 system: str,
                 phi: geod.Angle | None = None, 
                 g: float | None = None,
                 density: float = 2.67) -> None:
        match system:
            case 'orthometric':
                assert g is not None, 'Unspecified g in GeopotNum initialization'
                gr = g + dg_height(height/2) + 2*dg_bouguer(height/2, density)
                self.C = height * gr
            
            case 'dynamic':
                self.C = height * gamma0_grs80(geod.Angle(np.pi/4))
                
            case 'normal':
                assert phi is not None, 'Unspecified phi in GeopotNum initialization'
                gr = gamma0_grs80(phi) - dg_height(height/2)
                self.C = height * gr
                
            case 'geopotential':
                self.C = height * 1e6
                
            case _:
                raise TypeError('Invalid height system, read documentation dude')
                
    def __str__(self):
        return f'{self.C:.3f}'
                