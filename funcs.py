# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:20:25 2025

@author: pnapi

Oi, how'd u end up her ma'e?
"""

from . import geod
import numpy as np
from . import obj
from .const import a, e2, ro

def flh2XYZ(phi,la,h):
    """
    
    [φλh] -> [XYZ]
    Use Coordinates classes, this function only exists to make the class exist
    
    """
    x = round(((geod.find_N(obj.Angle(phi),a,e2)+h) * np.cos(phi) * np.cos(la)),3)
    y = round(((geod.find_N(obj.Angle(phi),a,e2)+h) * np.cos(phi) * np.sin(la)),3)
    z = round(((geod.find_N(obj.Angle(phi),a,e2)*(1-e2)+h)*np.sin(phi)),3)
    return x, y, z

def XYZ2flh(x,y,z,a=a,e2=e2):
    """
    
    [XYZ] -> [φλh]
    Use Coordinates classes, this function only exists to make the class exist
    
    """
    la = np.arctan2(y,x)
    radius = np.sqrt(x**2+y**2)
    phi = np.arctan2(z , radius*(1-e2))
    
    fs = 10**10 #random absurd value
    
    while abs(phi - fs) > (0.000001/ro):
        N = geod.find_N(obj.Angle(phi),a,e2)
        height = (radius/np.cos(phi) - N)
        fs = phi
        phi = np.arctan2(z , radius * (1 - (N*e2/(N+height))))
    return(phi,la,round(height,3))

def matrix_R_for_neu_transformation(crd):
    
    """
    Get the R matrix for neu coordinate translation.
    
    #--------------------------#

    Args:
        crd (geod.Coordinates(2d or 3d)):
            Coordinates at which R matrix is calculated.
        
        
    #--------------------------#

    Ret:
        R (np.array of size 3x3)
            R matrix
    """
    # if type(crd) is geod.obj.Coordinates3d:
    #     fi, la, h = crd.convert('flh')
    #     del h
    # elif type(crd) is geod.obj.Coordinates2d:
    #     fi, la = crd.convert('fl')
    # else: 
    #     raise geod.CoordinatesTypeError('crd must be a geod.Coordinates object')
    
    fi, la, h = crd.convert('flh')
    
    fi, la = fi.rad, la.rad
    
    n = [-np.sin(fi)*np.cos(la) , -np.sin(fi)*np.sin(la) , np.cos(fi)]
    e = [-np.sin(la) , np.cos(la) , 0]
    u = [np.cos(fi)*np.cos(la) , np.cos(fi)*np.sin(la) , np.sin(fi)]

    R = np.array([n, e, u])
    return R.T


def A0246(e2=e2):
    """
    Don't ask...'
    """
    A0 = 1-e2/4 - (3 *e2**2)/64 - (5*e2**3)/256
    A2 = (3/8) * (e2 + (e2**2)/4 + (15*e2**3)/128)
    A4 = (15/256) * (e2**2 + (3*e2**3)/4)
    A6 = (35*e2**3)/3072
    return A0, A2, A4, A6


def fi1(xgk, e2=e2):
    """
    Since most of GK algorithms are iteration based,
    calculation of the first approximation of latitude is needed.
    This function does that for you.
    """
    A0,A2,A4,A6 = A0246(e2=e2)
    fi = xgk/(a*A0)
    sigma =  a * (A0*fi - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))
    
    while True:
        prev_fi = fi
        fi = fi + (xgk-sigma)/(a*A0)
        sigma = a * (A0*fi - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))
        if abs(fi-prev_fi) < 0.000001/ro : break
    return fi

def lambda_0(la):
    return obj.Angle([round(la.rad / (2*np.pi) * 120) * 3], 'dms')

def fl2XY_GaussKruger(fi, la, l0, a=a, e2=e2):
    
    """
    [φλh] -> [XY_gk]
    Use Coordinates classes, this function only exists to make the class exist
    """
    
    # a - wielka półoś elipsoidy
    # b - mała półoś elipsoidy
    # e - pierwszy mimośród
    # e' - drugi mimośród
    # fi, la - współrzędne geodezyjne 
    # delta_la - różnica długości geodezyjnej
    # sigma - długość łuku południka
    # l0 - długość geodezyjne południka zerowego
    # t - tg(fi)
    # eta2 - e'2*cos2(fi)
    # N = a/(sqrt(1-e2sin2(fi)))

    
    a2 = a**2 
    b2 = a2*(1-e2)
    e2_prim = (a2-b2)/b2
    
    delta_la = la - l0
    t = np.tan(fi)
    eta2 = e2_prim * (np.cos(fi))**2
    # print(type(fi))
    if type(fi) == obj.Angle:
        N = geod.find_N(fi)
    else:
        N = geod.find_N(obj.Angle(fi))
    
    A0,A2,A4,A6 = A0246(e2=e2)
    sigma = a * (A0*fi.rad - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))
    
    xgk = sigma + ((delta_la.rad**2)/2) * N * np.sin(fi) * np.cos(fi) * (1+((delta_la.rad**2)/12) * (np.cos(fi))**2 * (5 -t**2 +9*eta2 + 4*eta2**2) + ((delta_la.rad**4)/360)* (np.cos(fi)**4) *(61 - 58*t**2 + t**4 + 270*eta2 - 330*eta2*t**2))
    ygk = delta_la.rad * N * np.cos(fi) * (1+((delta_la.rad**2)/6) * (np.cos(fi))**2 * (1 -t**2 +eta2) + ((delta_la.rad**4)/120)* (np.cos(fi)**4) *(5 - 18*t**2 + t**4 + 14*eta2 - 58*eta2*t**2))
    return xgk, ygk


def XY_GaussKruger_2fl(xgk, ygk, l0, a=a, e2=e2):
    """
    [XY_gk] -> [φλh] 
    Use Coordinates classes, this function only exists to make the class exist
    """
    
    A0,A2,A4,A6 = A0246(e2=e2)
    fi = xgk/(a*A0)
    sigma =  a * (A0*fi - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))
    
    while True:
        prev_fi = fi
        fi = fi + (xgk-sigma)/(a*A0)
        sigma = a * (A0*fi - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))
        if abs(fi-prev_fi) < 0.000001/ro : break

    a2 = a**2 
    b2 = a2*(1-e2)
    e2_prim = (a2-b2)/b2
    
    t = np.tan(fi)
    eta2 = e2_prim * (np.cos(fi))**2
    N = geod.find_N(obj.Angle(fi))
    M = geod.find_M(obj.Angle(fi))
    #jytdyjhmfujryt
    f = fi - (((ygk**2)*t)/(2*M*N)) * (1 - ((ygk**2)/(12*N**2)) * (5+3*t**2 + eta2 - 9*eta2*t**2-4*eta2**2) + (ygk**4)/(360*N**4) * (61+90*t**2+45*t**4))
    # la = l0 + (ygk/(N*np.cos(fi))) *   (1 - ((ygk**2)/( 6*N**2)) * (1+2*t**2 + eta2) + ((ygk**4)/(120*N**4)) * (5 + 28*t**2 + 24*t**4 + 6*eta2 + 8*eta2*t**2))
    try:
        l0 = l0.rad
    except AttributeError: pass
    # l0 = l0.rad
    la = (
    l0
    + (ygk / (N * np.cos(fi)))
    * (
        1
        - (ygk**2 / (6 * N**2)) * (1 + 2 * t**2 + eta2)
        + (ygk**4 / (120 * N**4)) * (5 + 28 * t**2 + 24 * t**4 + 6 * eta2 + 8 * eta2 * t**2)
    )
)
    return f, la


def fl2_PL2000(fi, la):
    """
    [φλh] -> [PL2000]
    Use Coordinates classes, this function only exists to make the class exist
    """
    m0 = 0.999923
    
    l0_num = round(lambda_0(la).rad * 60/np.pi)
       
    xgk, ygk = fl2XY_GaussKruger(fi, la, lambda_0(la))
    x2000 = xgk * m0
    y2000 = ygk * m0 + 500000.0 + 1000000.0*(l0_num) 
    return x2000, y2000
    
def PL2000_2fl(x2000, y2000):
    """
    [PL2000] -> [φλh]
    Use Coordinates classes, this function only exists to make the class exist
    """
    m0 = 0.999923
    xgk = x2000/m0
    yd = y2000%1000000
    l0_num = round((y2000 - yd)/1000000)
    ygk = (y2000 - 500000 - l0_num*1000000.0) / m0
    
    l0 = l0_num /60 * np.pi

    fi, la = XY_GaussKruger_2fl(xgk, ygk, l0)
    return obj.Angle(fi), obj.Angle(la)
    
def fl2_PL1992(fi, la):
    """
    [φλh] -> [PL1992]
    Use Coordinates classes, this function only exists to make the class exist
    """
    m0 = 0.9993
    l0 = 19/360 *2*np.pi
    xgk, ygk = fl2XY_GaussKruger(fi, la, obj.Angle(l0))
    x1992 = xgk * m0 - 5300000.0
    y1992 = ygk * m0 + 500000.0
    return x1992, y1992

def PL1992_2fl(x1992, y1992):
    """
    [PL1992] -> [φλh]
    Use Coordinates classes, this function only exists to make the class exist
    """
    l0 = obj.Angle([19])
    m0 = 0.9993
    xgk = (x1992 + 5300000) / m0
    ygk = (y1992 - 500000) / m0
    fi,la = XY_GaussKruger_2fl(xgk, ygk, l0)
    return fi, la

def PL1992_to_xyGK(x1992, y1992):
    """
    [PL1992] -> [XY_gk]
    Use Coordinates classes, this function only exists to make the class exist
    """
    f, l = PL1992_2fl(x1992, y1992)
    xgk, ygk = fl2XY_GaussKruger(f, l, obj.Angle(19, 'dms'))
    return xgk, ygk

def PL2000_to_xyGK(x2000, y2000):
    """
    [PL2000] -> [XY_gk]
    Use Coordinates classes, this function only exists to make the class exist
    """
    f, l = PL2000_2fl(x2000, y2000)
    xgk, ygk = fl2XY_GaussKruger(f, l, lambda_0(l))
    return xgk, ygk
