# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:34:18 2025

@author: pnapi

Gravimetry fraction of geod package
Contains Height and GeopotNum classes for precise height calculations
using gravity and stuff.

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
    
def gamma0_grs80(phi) -> float:
    gamma_0 = (978032.66 
               * (1 + 0.0053024*phi.sin()**2 - 0.0000058*phi.sin(2)**2))
    return (gamma_0)

def dg_bouguer(height: float, density: float = 2.67) -> float:
    """
    Returns dg_bouguer correction to gravimetry calculations.
    Unit: mGal
    """
    return round(-0.04192 * density * height, 4)

def dg_poincaryprey(height: float, density: float = 2.67) -> float:
    """
    Returns dg_poincaryprey correction to gravimetry calculations.
    Unit: mGal
    """
    return round((0.3086 - 2*0.04192 * density) * height, 4)

def anomaly(phi, g: float, dg: float) -> float:
    """
    Returns gravitational anomaly in a given point, in mGal

    Args:
        phi (geod.Angle):
            Point's latitude
        g (float):
            Gravitational acceleration in mGal
        dg (float): 
            dg correction (Height / Bouguer / Poincary-Prey)

    Returns:
        float: DESCRIPTION.

    """
    
    delta = g + dg - gamma0_grs80(phi)
    return round(delta, 4)
    
def height_dif_geopot(h_dif: float, g_a:float, g_b: float) -> float:
    """
    Adjusts the measured height difference between 2 points to the 
    geopotential height system.

    Args:
        h_dif (float): 
            Measured height difference between points
        g_a (float): 
            Gravitational acceleration measured on starting point in mGal
        g_b (float):
            Gravitational acceleration measured on ending point in mGal

    Returns:
        Height difference in geopotential system.
    """    
    gi = np.average([g_a, g_b])
    return h_dif + (gi - 1e6)/1e6 * h_dif
    
def height_dif_dynamic(h_dif: float, g_a:float, g_b: float) -> float:
    """
    Adjusts the measured height difference between 2 points to the 
    dynamic height system.

    Args:
        h_dif (float): 
            Measured height difference between points
        g_a (float): 
            Gravitational acceleration measured on starting point in mGal
        g_b (float):
            Gravitational acceleration measured on ending point in mGal

    Returns:
        Height difference in dynamic system.
    """
    gi = np.average([g_a, g_b])
    g45 = gamma0_grs80(obj.Angle([45]))
    return h_dif + (gi - g45)/g45 * h_dif    
    
def height_dif_orthometric(H_appr_A: float,
                           H_appr_B: float,
                           h_dif: float,
                           g_a: float,
                           g_b: float) -> float:
    """
    Adjusts the measured height difference between 2 points to the 
    orthometric height system.

    Args:
        H_appr_A (float):
            Approximate (or exact if there) height of starting point 
        H_appr_B (float):
            Approximate (or exact if there) height of ending point 
        h_dif (float): 
            Measured height difference between points
        g_a (float): 
            Gravitational acceleration measured on starting point in mGal
        g_b (float):
            Gravitational acceleration measured on ending point in mGal

    Returns:
        Height difference in orthometric system.
    """
    g045 = gamma0_grs80(obj.Angle(np.pi/4))
    gr_a = g_a + dg_height(H_appr_A/2) + 2*dg_bouguer(H_appr_A/2)
    gr_b = g_b + dg_height(H_appr_B/2) + 2*dg_bouguer(H_appr_B/2)
    gi = np.average([g_a, g_b])
    return ((gi-g045) / g045 * h_dif
            + (gr_a - g045) / g045 * H_appr_A
            - (gr_b - g045) / g045 * H_appr_B) + h_dif
    
def height_dif_normal(phi_a, phi_b,
                      H_appr_A: float, H_appr_B: float, 
                      h_dif: float, 
                      g_a: float, g_b: float) -> float:
    """
    Adjusts the measured height difference between 2 points to the 
    normal height system.

    Args:
        phi_a (geod.Angle):
            Latitude of starting point
        phi_b (geod.Angle):
            Latitude of ending point
        H_appr_A (float):
            Approximate (or exact if there) height of starting point 
        H_appr_B (float):
            Approximate (or exact if there) height of ending point 
        h_dif (float): 
            Measured height difference between points
        g_a (float): 
            Gravitational acceleration measured on starting point in mGal
        g_b (float):
            Gravitational acceleration measured on ending point in mGal

    Returns:
        Height difference in normal system.
    """
    
    gi = np.average([g_a, g_b])
    g045 = gamma0_grs80(obj.Angle([np.pi/4]))
    gam_a = gamma0_grs80(phi_a) - 0.3086*(H_appr_A/2)
    gam_b = gamma0_grs80(phi_b) - 0.3086*(H_appr_B/2)
    return ((gi-g045) / g045 * h_dif
            + (gam_a - g045) / g045 * H_appr_A
            - (gam_b - g045) / g045 * H_appr_B) + h_dif
    
def height_geopot(C):
    """
    Returns height in geopotential system.

    Args:
        C (float | geod.grav.GeopotNum):
            Geopotential number.

    Returns:
        Height in geopotential system.

    """
    if hasattr(C, 'C'): C = C.C
    return C / 10**6

def height_dynamic(C):
    """
    Returns height in dynamic system.

    Args:
        C (float | geod.grav.GeopotNum):
            Geopotential number.

    Returns:
        Height in dynamic system.

    """
    if hasattr(C, 'C'): C = C.C
    return C / gamma0_grs80(obj.Angle(np.pi/4))

def height_orthometric(C, g, H_appr):
    """
    Returns height in geopotential system.

    Args:
        C (float | geod.grav.GeopotNum):
            Geopotential number.
        g (float):
            Gravitational acceleration in mGal
        H_appr (float):
            Approximate height of the point

    Returns:
        Height in geopotential system.

    """
    if hasattr(C, 'C'): C = C.C
    g = g + dg_poincaryprey(H_appr/2) 
    return C / g

def height_normal(C, phi, H_appr):
    """
    Returns height in geopotential system.

    Args:
        C (float | geod.grav.GeopotNum):
            Geopotential number.
        phi (geod.Angle):
            Latitude of the point
        H_appr (float):
            Approximate height of the point

    Returns:
        Height in geopotential system.

    """
    if hasattr(C, 'C'): C = C.C
    g = gamma0_grs80(phi) - dg_height(H_appr/2) 
    return C / g


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
        H_appr (float): 
            Approximate height of the point, 
            needed for gravity stuff calculations, i don even know bruh
    """
    def __init__(self, 
                 C: float, 
                 output_type: str | None,
                 phi = None, 
                 g: float | None = None,
                 H_appr: float | None = None) -> None:
        if hasattr(C, 'C'):
            self.C = C.C 
        else:
            self.C = C
        
        self.phi = phi
        self.H_appr = H_appr
        self.g = g
        
        self.output_type = output_type
        
    def convert(self, output_type: str) -> float:
        
        match output_type:
            case 'orthometric':
                assert self.g is not None, 'Unspecified g in Height initialization'
                assert self.H_appr is not None, 'Unspecified H_appr in Height initialization'
                return height_orthometric(self.C, self.g, self.H_appr)
            
            case 'dynamic':
                return height_dynamic(self.C)
            
            case 'normal':
                assert self.phi is not None, 'Unspecified phi in Height initialization'
                assert self.H_appr is not None, 'Unspecified H_appr in Height initialization'
                return height_normal(self.C, self.phi, self.H_appr)
                
            case 'geopotential':
                return height_geopot(self.C)
            
            case _:
                raise TypeError('Invalid output_type, read documentation dude')
                
    def __str__(self) -> str:
        return f'{self.convert(self.output_type):.3f}'
    
    def __repr__(self) -> str:
        return f'{self.convert(self.output_type):.3f}'
    

class GeopotNum:
    """
    Calculate a point's geopotential number
    without dealing with all pesky stuff.

    Args:
        height (float): 
            Point's height in m
        system (str): 
            Select the height system in which the height is given.
            'normal / orthometric / dynamic / geopotential'   
            
        These next ones are not mandatory, used only for certain systems,
        you'll be warned if you miss them dw pookie
        
        phi (geod.Angle): 
            Latitude of the point where GeopotNum is calculated            
        g (float): 
            Gravitational acceleration, in units of mGal.
        density (float):
            Land mass density below the point, 
            defaults to 2.67g/cm3 being the avg density of Earth
    """
    def __init__(self, height: float,
                 system: str,
                 phi = None, 
                 g: float | None = None,
                 density: float = 2.67) -> None:
        match system:
            case 'orthometric':
                assert g is not None, 'Unspecified g in GeopotNum initialization'
                gr = g + dg_height(height/2) + 2*dg_bouguer(height/2, density)
                self.C = height * gr
            
            case 'dynamic':
                self.C = height * gamma0_grs80(obj.Angle(np.pi/4))
                
            case 'normal':
                assert phi is not None, 'Unspecified phi in GeopotNum initialization'
                gr = gamma0_grs80(phi) - dg_height(height/2)
                self.C = height * gr
                
            case 'geopotential':
                self.C = height * 1e6
                
            case _:
                raise TypeError('Invalid height system, read documentation dude')
                
    def __str__(self):
        'print me daddy uwu'
        return f'{self.C:.3f}'
    
    def __repr__(self):
        'represent me daddy uwu'
        return f'{self.C:.3f}'

                