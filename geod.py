# -*- coding: utf-8 -*-
"""
@author: pnapi
v.1.0

#-------------------------------------------------------------#

This module contains some of the geodesy-related stuff.
Maybe you'll find something useful...

Run geod.info() for... well, info.

"""


#-------------------------------------------------------------#


from .const import a, e2, ro, rho, m0_1992, m0_2000
import numpy as np
from . import obj
from . import funcs as fn


#-------------------------------------------------------------#


class StupidError(ValueError):
    """
    Definition of an error.
    Do I really have to explain this...
    If you somehow get this error - reconsider your life choices.
    """


class CoordinatesTypeError(ValueError):
    """
    Raised when trying to perform calculations on/to/from a coordinate system
    invalid in that case.
    """
    

class MissingArgError(ValueError):
    """
    Just a simple definition of an error message, nothing special to see here.
    """
    

#-------------------------------------------------------------#

    
def approx(num,rnd):
    'Broken, use built-in round, fix in progress'
     
    # raise AttributeError('geod.approx is broken, use built-in round method, fix in progress')
    return round(num, rnd)
    
    # """    
    # Float number rounding
    # the geodetic way.
    
    # #--------------------------#

    # Args:
    #     num (float):
    #         Number to round
    #     rnd (int):
    #         Amount of decimal places after the coma,
    #         (obviously must be a positive number)
        
    # #--------------------------#

    # Returns:
    #     Geodetically rounded number. (god i hope i didn't butcher my english here)
    # """
    if rnd < 0:
        raise StupidError(
            'Bruh, you\'re trying to round to negative number of digits?')
    if type(rnd) is not int:
        raise StupidError(
            'Bruh, you\'re really trying to round to a float number of digits?')
        
    if type(num) not in (float, int, np.float64):
        raise TypeError('Invalid variable type to round')
        
    # yes i know this approach is heathenly and ungodly,
    # don't blame me i was tired
    # and yes, pep-8 is explicitly bumbutchered here
    # hey ya, new words ^.^
    
    w, dec = str(num).split('.')
    # print(w,dec)
    if int(dec[rnd]) == 5:
        # print()
        if int(dec[rnd-1])%2 == 0:
            dec = dec[:rnd]
        else:
            dec = dec[:(rnd-1)] + str(int(dec[rnd-1])+1)
    elif int(dec[rnd]) < 5:
        dec = dec[:rnd]
    else:
        dec = dec[:(rnd-1)] + str(int(dec[rnd-1])+1)
    
    return float(w + '.' + dec)


def line_length(a, b, crd_type):
    """
    

    Args:
        a (geod.Coordinates(3d or 2d)): first point of a line.
        b (geod.Coordinates(3d or 2d)): second point of a line.
        crd_type (str): 
            Coordinate system in which you want the line length to be calculated.

    Returns:
        s (float): Calculated line length.

    """
    
    # a & b must be coordinates defined as a coordinate object
    # they'll be automatically taken care of in whatever coords system you wish
    # however it doesn't work for φλh so they have to be converted to XYZ
    if crd_type in ['fl', 'flh']:
        raise CoordinatesTypeError(f'Convert {crd_type} to ortho first,'
                                   + 'or use geod.orthodromic() instead')
    
    try:
        a = a.convert(crd_type)
        b = b.convert(crd_type)
    except AttributeError:
        raise TypeError('Are you sure you\'re using the right variable?')
    
    # calculate differences
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    # try -> if 3D just do it, 
    # else -> just ignore and make it a zero
    try:
        dz = b[2] - a[2]
    except IndexError:
        dz = 0
    
    # just classic pythagorean
    s = np.sqrt(dx**2 + dy**2 + dz**2)
    return s


def find_N(phi, a=a, e2=e2):
    """
    
    Arg:
        phi (geod.Angle)
            Latitude (φ)
        a (float):
            The default is geod.a.
        e2 (float):
            The default is geod.e2.

    #--------------------------#
    
    Ret:
        N (float):
            Maximal curvature radius in a given latitude (φ).

    """
    N = a / (np.sqrt(1-e2*(np.sin(phi.rad))**2))
    return round(N, 5)

def find_M(phi, a=a, e2=e2):
    
    """
    
    Arg:
        phi (geod.Angle)
            Latitude (φ)
        a (float):
            The default is geod.a.
        e2 (float):
            The default is geod.e2.

    #--------------------------#
    
    Ret:
        M (float):
            Minimal curvature radius in a given latitude (φ).

    """
    
    M = (a * (1 - e2)) / (np.sqrt((1-e2*(np.sin(phi.rad))**2)**3))
    return approx(M, 5)
    
def find_R(phi, a=a, e2=e2):
    """
    
    Arg:
        phi (geod.Angle)
            Latitude (φ)
        a (float):
            The default is geod.a.
        e2 (float):
            The default is geod.e2.

    #--------------------------#
    
    Ret:
        R (float):
            Average curvature radius in a given latitude (φ).

    """
    
    return approx(np.sqrt(find_M(phi, a, e2) * find_N(phi, a, e2)), 5)

def get_orthodromic_end(crd1, azim_AB, length_of_line, a=a, e2=e2):
    """
    
    Calculates the azimuth from second point to the first
    and the coordinates of the second point.
    
    #--------------------------#
    
    Args: 
        crd1 (geod.Coordinates(3d or 2d)):
            Coordinates of the first point of the line.
        azim_AB (geod.Angle):
            Azimuth A-B on the orthodromic line.
        length_of_line (float):
            Length of the line.
        a (float):
            The default is geod.a.
        e2 (float):
            The default is geod.e2.
            
    #--------------------------#

    Returns:
        azim_BA (geod.Angle):
            Azimuth from end to start of the orthodromic line.         
        b (geod.Coordinates2d):
            Coordinates of the end of orthodromic line.
        
    """
    if length_of_line<=0:
        raise StupidError(
            f'Yeah, just think about it... length_of_line = {length_of_line}')
    
    rg = int(length_of_line/1000)
    section_length = length_of_line/rg
    
    if type(crd1) == obj.Coordinates3d:
        phi, la, h = crd1.convert('flh')
    elif type(crd1) == obj.Coordinates2d:
        phi, la = crd1.convert('fl')
    else: 
        raise CoordinatesTypeError('crd1 must be a geod.Coordinates object')
        
    azim_AB, phi, la = azim_AB.rad, phi.rad, la.rad    
    
    for _ in range(rg):
        N = find_N(obj.Angle(phi),a,e2)
        M = find_M(obj.Angle(phi),a,e2)
        
        d_phi = section_length * np.cos(azim_AB) / M
        d_azim_AB = section_length * np.sin(azim_AB) * np.tan(phi) / N
        
        phi_mi = phi + d_phi/2 
        azim_AB_mi = azim_AB + d_azim_AB/2 
        
        N_mi = find_N(obj.Angle(phi_mi), a, e2)
        M_mi = find_M(obj.Angle(phi_mi), a, e2)
        
        d_phi_mi = section_length * np.cos(azim_AB_mi) / M_mi
        d_azim_AB_mi = section_length * np.sin(azim_AB_mi) * np.tan(phi_mi) / N_mi
        d_la_mi = section_length * np.sin(azim_AB_mi) / (N_mi * np.cos(phi_mi))
        
        phi = phi + d_phi_mi
        la = la + d_la_mi
        
        azim_AB = azim_AB + d_azim_AB_mi
    
    azim_AB = azim_AB + np.pi
    
    if (azim_AB > 2*np.pi):
        azim_BA = obj.Angle(azim_AB - 2*np.pi)
    else:
        azim_BA = obj.Angle(azim_AB)
        phi_b , la_b = phi , la
    
    b = obj.Coordinates2d('fl', phi_b, la_b)
    
    return azim_BA, b

def orthodromic(crd1, crd2, a=a, e2=e2):
    
    """
    
    Algorytm Vincentego
    
    #--------------------------#
    
    Args: 
        crd1 (geod.Coordinates(2d or 3d)):
            Coordinates of the first end of the line.
        crd2 (geod.Coordinates(2d or 3d)):
            Coordinates of the second end of the line.
        a (float):
            The default is geod.a.
        e2 (float):
            The default is geod.e2.
                
    #--------------------------#

        Returns:
            s : float
                Dł. linii geodezyjnej
            azim_AB : float (w radianach)
                Azymut na linii geodezyjnej z punktu A na B 
            azim_BA : float (w radianach)
                Azymut na linii geodezyjnej z punktu B na A 
    """
    
    # a - wielka półoś elipsoidy
    # b - mała półoś elipsoidy
    # e - pierwszy mimośród
    # f - spłaszczenie elipsoidy
    # fi, la - współrzędne geodezyjne 
    # delta_la - różnica długości geodezyjnej
    # s - długość linii geodezyjnej
    # azim_AB, azim_BA - azymut prosty i odwrotny
    # alfa - azymut linii geodezyjnej na równiku
    # U_a, U_b - szer. zredukowane
    # L - różnica dł. na sferze zredukowanej
    # sigma - odl. kątowa pomiędzy punktami na sferze
    # sigma_m - odl. kątowa na sferze od równika do punktu środkowego l. geodezyjnej
    
    print(type(crd1))
    if type(crd1) == obj.Coordinates3d:
        phi_a, la_a, ha = crd1.convert('flh')
    elif type(crd1) is obj.Coordinates2d:
        phi_a, la_a = crd1.convert('fl')
    # else: 
    #     raise CoordinatesTypeError('crd1 must be a geod.Coordinates object')
        
    if type(crd2) == obj.Coordinates3d:
        phi_b, la_b, hb = crd2.convert('flh')
    elif type(crd2) == obj.Coordinates2d:
        phi_b, la_b = crd2.convert('fl')
    # else: 
    #     raise CoordinatesTypeError('crd1 must be a geod.Coordinates object')
    
    phi_a, phi_b, la_a, la_b = phi_a.rad, phi_b.rad, la_a.rad, la_b.rad
    
    try:
        del ha
        del hb
    except NameError:
        pass
    
    #krok 1
    b = a * (np.sqrt(1-e2))
    f = 1 - (b/a)
    
    #krok 2
    delta_la = la_b - la_a
    
    #krok 3
    U_a = np.arctan((1 - f) * np.tan(phi_a))
    U_b = np.arctan((1 - f) * np.tan(phi_b))
    
    #krok 4
    L = delta_la
    
    while(True):
        
        prev_L = L
        
        #krok 5
        sin_sigma = np.sqrt((np.cos(U_b) * np.sin(L))**2 + (np.cos(U_a) * np.sin(U_b) - np.sin(U_a) * np.cos(U_b) * np.cos(L))**2)

        #krok 6 
        cos_sigma = np.sin(U_a) * np.sin(U_b) + np.cos(U_a) * np.cos(U_b) * np.cos(L)
        #print(sin_sigma,cos_sigma)
        #krok 7
        sigma = np.arctan2(sin_sigma, cos_sigma)
        
        #krok 8
        sin_alfa = (np.cos(U_a) * np.cos(U_b) * np.sin(L)) / sin_sigma
        
        #krok 9
        cos2_alfa = 1 - (sin_alfa**2)
        
        #krok 10
        cos_2sigma_m = cos_sigma - ((2 * np.sin(U_a) * np.sin(U_b)) / cos2_alfa)
        
        #krok 11
        C = f/16 * cos2_alfa * (4 + f *(4 - 3*cos2_alfa))
        
        #krok 12
        L = delta_la + (1-C)* f * sin_alfa * (sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1 + 2* (cos_2sigma_m**2))))        

        #czy wyjść z pętli?
        if (abs(L - prev_L) < (10**-6/ro)):
            break
        
    #krok 13
    u2 = (a**2 - b**2) / b**2 * cos2_alfa 
    
    #krok 14
    A = 1 + (u2 / 16384)* (4096 + u2*(-768 + u2*(320 - 175*u2)))
    
    #krok 15
    B = (u2 / 1024)* (256 + u2* (-128 + u2*(74 - 47*u2)))
    
    #krok 16 (zmienne pomocnicze h służą rozbiciu równania na prostsze składowe)
    h0 = B*sin_sigma
    h1 = B*cos_2sigma_m
    h2 = -1 + 2* (cos_2sigma_m**2)
    h3 = -3 + 4* (sin_sigma**2)
    h4 = -3 + 4* (cos_2sigma_m**2)
    h5_sq_par = cos_sigma * h2 - (1/6)* h1 * h3 * h4
    h6_wav_par = cos_2sigma_m + (1/4) * B * h5_sq_par
    
    delta_sigma = h0 * h6_wav_par
    
    #krok 17
    s = b * A *(sigma - delta_sigma)
    
    #krok 18
    azim_AB = np.arctan2(np.cos(U_b) * np.sin(L) , np.cos(U_a) * np.sin(U_b) - np.sin(U_a) * np.cos(U_b) * np.cos(L))
    if azim_AB < 0:
        azim_AB += 2*np.pi
    #krok 19
    azim_BA = np.arctan2(np.cos(U_a) * np.sin(L) , -np.sin(U_a) * np.cos(U_b) + np.cos(U_a) * np.sin(U_b) * np.cos(L)) + np.pi
    
    return s, obj.Angle(azim_AB), obj.Angle(azim_BA)

def neu2XYZ(delta_x_neu, crd):
    
    """
    Vector conversion - Δneu -> ΔXYZ
    
    #--------------------------#

    Args:
        delta_x_neu (np.array of size 3x1):
            Vector of Δneu
        crd (geod.Coordinates(2d or 3d)):
            Coordinates at which conversion is performed.
        
    #--------------------------#

    Returns:
        delta_X_xyz (np.array of size 3x1):
            Vector of ΔXYZ
    """
    R = fn.matrix_R_for_neu_transformation(crd)
    delta_X_xyz = R @ delta_x_neu
    return delta_X_xyz

def XYZ2neu(delta_X_xyz, crd):
    
    """
    Vector conversion - ΔXYZ -> Δneu
    
    #--------------------------#

    Args:
        delta_X_xyz (np.array of size 3x1):
            Vector of ΔXYZ
        crd (geod.Coordinates(2d or 3d)):
            Coordinates at which conversion is performed. 
        
    #--------------------------#

    Returns:
        delta_x_neu (np.array of size 3x1):
            Vector of Δneu
    """
    R = fn.matrix_R_for_neu_transformation(crd) 
    delta_X_neu = R.T @ delta_X_xyz
    return delta_X_neu

def get_neu_delta_vect(vect_length, zenith, alpha):
    
    """
    Converts observation data to neu coords translation vector.
    
    #--------------------------#

    Arg:
        vect_length (float):
            Length of the spatial vector between 2 points
        zenith : (geod.Angle):
            Vertical zenithal angle observed
        alpha (geod.Angle):
            Horizontal α angle observed
        
    #--------------------------#

    Returns:
        delta_x_neu (np.array of size 3x1):
            Vector of Δneu
    """
    if vect_length < 0:
        raise StupidError('Really? vect_length = '+str(vect_length))
    dn = vect_length * np.sin(zenith.rad) * np.cos(alpha.rad)
    de = vect_length * np.sin(zenith.rad) * np.sin(alpha.rad)
    du = vect_length * np.cos(zenith.rad)
    
    neu_delta_vect = np.array([[dn],
                           [de],
                           [du]])
    return neu_delta_vect


def get_obs_data_from_neu(neu):
    """
    Converts neu coords translation vector to observation data.
    
    #--------------------------#

    Args:
        delta_x_neu (np.array of size 3x1):
            Vector of Δneu
                
    #--------------------------#

    Returns:
        vect_length (float):
            Length of the spatial vector between 2 points
        zenith : (geod.Angle):
            Vertical zenithal angle observed
        alpha (geod.Angle):
            Horizontal α angle observed
    """
    assert neu.shape == (3,1) or len(neu) == 3, f'neu vector must be of size (3) or (3,1), not {neu.shape()}'
    try:
        n = neu[0][0]
        e = neu[1][0]
        u = neu[2][0]
    except IndexError:
        n = neu[0]
        e = neu[1]
        u = neu[2]
    s = np.sqrt(n**2+e**2+u**2)
    alpha = np.arctan2(e,n)
    z = np.arccos(u/s)
    return s, obj.Angle(z), obj.Angle(alpha)

    
def projection_scale_GK(crd, la_0, a=a, e2=e2):
    """
    Returns a GK projection scale at a given point
    
    Args:
        crd (geod.Coordinates(2d or 3d)):
            Coordinates
        la_0 (geod.Angle):
            Lambda 0 - Reference longitude for GK conversion
        e2 (float):
            The default is geod.e2.
            
    Returns:
        mgk (float):
            Projection scale
    """

    xgk, ygk = crd.convert('gk', la_0=la_0)

    A0, A2, A4, A6 = fn.A0246(e2=e2)
    fi = xgk/(a*A0)
    sigma = a * (A0*fi - A2*np.sin(2*fi) + A4*np.sin(4*fi) - A6*np.sin(6*fi))

    while True:
        prev_fi = fi
        fi = fi + (xgk-sigma)/(a*A0)
        sigma = a * (A0*fi - A2*np.sin(2*fi) + A4 *
                     np.sin(4*fi) - A6*np.sin(6*fi))
        if abs(fi-prev_fi) < 0.000001/rho:
            break

    a2 = a**2
    b2 = a2*(1-e2)
    e2_prim = (a2-b2)/b2

    t = np.tan(fi)
    eta2 = e2_prim * (np.cos(fi))**2
    N = find_N(fi)
    M = find_M(fi)

    fi1 = fi - (((ygk**2)*t)/(2*M*N)) * (1 - ((ygk**2)/(12*N**2)) * (5+3*t**2 +
                                                                     eta2 - 9*eta2*t**2-4*eta2**2) + (ygk**4)/(360*N**4) * (61+90*t**2+45*t**4))

    R = find_R(fi1)

    mgk = 1 + ((ygk**2)/(2*R**2)) + ((ygk**4)/(24*R**4))
    return mgk

def line_red_ell_to_GK(sgk, crd1, crd2, la_0, a=a, e2=e2):
    """
    I'm really really not in the mood to explain this...
    """
    
    xagk, yagk = crd1.convert('gk', la_0=la_0)
    xbgk, ybgk = crd2.convert('gk', la_0=la_0)
    
    assert sgk >= 0, 'Długość odcinka ujemna'
    
    xm, ym = (xagk+xbgk)/2 , (yagk+ybgk)/2
    fim, lam = fn.XY_GaussKruger_2fl(xm, ym, la_0)
    Rm = find_R(fim)
    
    red = sgk * ((yagk**2 + yagk*ybgk + ybgk**2)/(6* Rm**2))
    return red

def line_len_ell_to_GK(s_elip, crd1, crd2, la_0, a=a, e2=e2):
    """
    I'm really really not in the mood to explain this...
    """
    
    xagk, yagk = crd1.convert('gk', la_0=la_0)
    xbgk, ybgk = crd2.convert('gk', la_0=la_0)
    
    assert s_elip >= 0, 'Długość odcinka ujemna'
    red = line_red_ell_to_GK(s_elip, xagk, yagk, xbgk, ybgk, la_0)
    sgk = s_elip + red
    return sgk


def line_len_parallel_projection(s_measured, Rm, height_a, height_b):
    """
    Returns the length of a spatial vector 
    projected to the surface of the ellipsoid.
    """
    assert s_measured >= 0, 'Długość odcinka ujemna'
    ha, hb = height_a, height_b
    s0squ = (s_measured**2 - (ha-hb)**2) / ((1+ha/Rm) * (1+hb/Rm))
    s_elip = 2*Rm* np.arcsin((np.sqrt(s0squ))/(2*Rm))
    return s_elip

def convergence(crd, la_0, a=a, e2=e2):
    try:
        la_0 = la_0.rad
    except AttributeError: pass
    xgk, ygk = crd.convert('gk', la_0=la_0)
    la_0 = la_0
    
    a2 = a**2 
    b2 = a2*(1-e2)
    e2_prim = (a2-b2)/b2
    
    f1 = fn.fi1(xgk)
    try:
        fi, la, h = crd.convert('flh')
        del h
    except TypeError: 
        fi, la = crd.convert('fl')
            
    
    t = np.tan(f1)
    eta2 = e2_prim * (np.cos(fi.rad))**2
    N = find_N(fi)
    
    gam = (ygk/N)*t * (1- (ygk**2/(3*N**2)) * (1+t**2 - eta2 - 2*eta2**2) + ((ygk**4)/(15*N**4)) * (2+5*t**2 + 3*t**4))
    return obj.Angle(gam)

def ang_red(crd1, crd2, la_0):
    try:
        la_0 = la_0.rad
    except AttributeError: pass
    xagk, yagk = crd1.convert('gk', la_0=la_0)
    xbgk, ybgk = crd2.convert('gk', la_0=la_0)
    
    xm, ym = (xagk+xbgk)/2 , (yagk+ybgk)/2
    # fim, lam = fn.XY_GaussKruger_2fl(xm, ym, la_0)
    fim, lam = obj.Coordinates2d('gk', xm, ym, la_0=la_0).convert('fl')
    Rm = find_R(fim)
    
    dab = ((xbgk-xagk)*(2*yagk+ybgk))/(6*Rm**2)
    # dba = ((xagk-xbgk)*(2*ybgk+yagk))/(6*Rm**2)
    
    return obj.Angle(dab) #, dba

def find_azim(xa, ya, xb, yb):
    'Bobux generator placeholder?'
    dx = xb-xa
    dy = yb-ya
    return np.arctan2(dy,dx)

def azim_red(crd1, crd2, la_0):
    
    la_0 = la_0.rad
    xagk, yagk = crd1.convert('gk', la_0=la_0)
    xbgk, ybgk = crd2.convert('gk', la_0=la_0)
    
    alpha = obj.Angle(np.arctan2(ybgk-yagk, xbgk-xagk))
    gamma = convergence(crd1, la_0)
    delta = ang_red(crd1, crd2, la_0)
    
    return alpha+gamma+delta

#transformacje
def transMatrixA6(crd1):
    x1, y1, z1 = crd1.convert('xyz')
    return np.array([
        [  0, -z1,  y1, 1, 0, 0],
        [ z1,   0, -x1, 0, 1, 0],
        [-y1,  x1,   0, 0, 0, 1]
        ])


def transMatrixA7(crd1):
    x1, y1, z1 = crd1.convert('xyz')
    return np.array([
        [x1,   0, -z1,  y1, 1, 0, 0],
        [y1,  z1,   0, -x1, 0, 1, 0],
        [z1, -y1,  x1,   0, 0, 0, 1]
        ])

def transMatrixA9(crd1):
    x1, y1, z1 = crd1.convert('xyz')
    return np.array([
        [x1, 0, 0,   0, -z1,  y1, 1, 0, 0],
        [0, y1, 0,  z1,   0, -x1, 0, 1, 0],
        [0, 0, z1, -y1,  x1,   0, 0, 0, 1]
        ])

def transMatrixL(crd1, crd2):
    x1, y1, z1 = crd1.convert('xyz')
    x2, y2, z2 = crd2.convert('xyz')
    return np.array([
        [x2-x1],
        [y2-y1],
        [z2-z1]
        ])

def transParams(A, L):
    x = (np.linalg.inv(A.T @ A)) @ (A.T @ L)
    h = x.T[0]
    (kappa, alpha, beta, gamma, x0, y0, z0) = tuple(h)
    return kappa, alpha, beta, gamma, x0, y0, z0

# def bursaWolfTransform(crd1, crd0, alpha, beta, gamma, kx, ky=None, kz=None):
#     x1, y1, z1 = crd1.convert('xyz')
#     x0, y0, z0 = crd0.convert('xyz')
    
#     alpha = alpha.rad
#     beta = beta.rad
#     gamma = gamma.rad
    
#     if kz == None and ky == None: 
#         ky = kx
#         kz = kx
#     xyz1 = np.array([[x1, y1, z1]]).T
#     B = np.array([
#         [  kx  ,  gamma, -beta],
#         [-gamma,   ky  , alpha],
#         [  beta, -alpha,   kz ]])
#     xyz0 = np.array([[x0,y0,z0]]).T
#     return xyz1 + (B @ xyz1) + xyz0

####################################################################
################## GW 2 --- where boys become men ##################
####################################################################

def centrif_acc(latitude: obj.Angle, 
                R: float=6_380_000, 
                rot_time: float=86400):
    return (2*np.pi / rot_time)**2 * R * latitude.cos()
    

def get_dist_azim_elevation(obs_crd, aim_crd):
    """
    Gives you the distance (in meters), azimuth and elevation angle
    from one point (observation) to another (aim).

    Args:
        obs_crd (geod.Coordinates3d):
            Coordinates of the first point 
            (where you want to get the observation data)
        aim_crd (geod.Coordinates3d):
            Coordinates of the other point
            (where you aim at)

    Returns:
        s (float):
            Distance between points, in meters
        azim (geod.Angle):
            Azimuth to the point aimed at.
        elev (geod.Angle):
            Elevation angle to that point.

    """
    neu=XYZ2neu(aim_crd-obs_crd, obs_crd)
    s, azim, elev = get_obs_data_from_neu(neu)
    return s, azim, elev

def niv_height_dif(w1, p1, p2, w2, niv_type):
    """
    Calculates height difference at a single difference measure.

    Args:
        w1 (int): 
            Younger backwards measure
        p1 (int): 
            Younger forward measure
        p2 (int): 
            Older forward measure
        w2 (int): 
            Older backwards measure
        niv_type:
            If Ni002, Ni004, Ni007 -> set value to 1
            If WildN3, OptionN1 -> set value to 0.5

    Returns:
        Height difference

    """
    assert niv_type in [1, 0.5], """niv_type:
        If Ni002, Ni004, Ni007 -> set value to 1
        If WildN3, OptionN1 -> set value to 0.5"""
    for i in (w1, p1, p2, w2):
        if not isinstance(i, int):
            raise TypeError("Measure can't be a float")
    return ((w1-p1) + (w2-p2)) / 2 * niv_type*1e-5

def quasihorizon_check(dh_cen, dh_ex, len_ex_p, len_ex_w):
    """
    Check whether quasihorizon error is within acceptable range.

    Args:
        dh_cen (float): 
            Height dif measured from central position
        dh_ex (float): 
            Height dif measured from excentric position
        len_ex_p (float): 
            Distance from the excentric position to forward point
        len_ex_w (float): 
            Distance from the excentric position to backwards point

    Returns:
        bool: Acceptable?
        geod.Angle: Error value
    """
    alpha = obj.Angle(np.arctan((dh_cen-dh_ex) / (len_ex_p-len_ex_w)))
    return alpha < obj.Angle([0,0,5]), alpha
    
def niv_correction_temp(dh, exp, temp_avg, temp_comp):
    """
    Returns temperature correction for leveling measures

    Args:
        dh (float): 
            Measured height difference
        exp (float): 
            Average expansion coefficient of two leveling staffs
            Unit: μm / (m * °C)
        temp_avg (float): 
            Average temperature in Celsius
        temp_comp (float): 
            Comparison temperature in Celsius
    """
    return dh * exp * (temp_avg - temp_comp) 

def niv_correction_comp(dh, meter_corr):
    """
    Returns comparison correction for leveling measures

    Args:
        dh (float): 
            Measured height difference
        meter_corr (float): 
            Average correction for average length of a leveling staff pair meter
    """
    return dh * meter_corr

def niv_correction_lunisolar(azim_to_object,
                              azim_of_line,
                              zenith_ang_to_object,
                              path_length,
                              object_type):
    """
    Returns lunisolar correction for leveling measures

    Args:
        azim_to_object (geod.Angle):
            Azimuth to the Sun or Moon
        azim_of_line (geod.Angle): 
            Azimuth of the measured line, from starting point
        zenith_ang_to_object (geod.Angle): 
            Zenithal distance of the Sun or Moon
        path_length (float):
            Length of the leveling path in meters
        object_type (str): 
            What to count the correction for?
            options -> ('moon', 'sun', 'both')
    """
    match object_type:
        case 'moon':
            k = 8.5e3
            return (0.7 * k * path_length * zenith_ang_to_object.sin(2) 
                    * (azim_to_object - azim_of_line).cos())
        case 'sun':
            k = 3.9e3
            return (0.7 * k * path_length * zenith_ang_to_object.sin(2) 
                    * (azim_to_object - azim_of_line).cos())
        # case 'both':
        #     return ((0.7 * 8.5e3 * path_length * zenith_ang_to_object.sin(2) 
        #             * (azim_to_object - azim_of_line).cos())
        #             + (0.7 * 3.9e3 * path_length * zenith_ang_to_object.sin(2) 
        #             * (azim_to_object - azim_of_line).cos()))
        
def height_dif_corrected(dh: float, 
                         exp: float, 
                         temp_avg: float, 
                         temp_comp: float, 
                         meter_corr: float, 
                         azim_to_moon: obj.Angle, 
                         zenith_ang_to_moon: obj.Angle, 
                         azim_to_sun: obj.Angle, 
                         zenith_ang_to_sun: obj.Angle, 
                         azim_of_line: obj.Angle, 
                         path_length: float):
    """
    Calculate height difference with all corrections included.

    Args:
        dh (float): 
            Measured height difference
        exp (float): 
            Average expansion coefficient of two leveling staffs
            Unit: μm / (m * °C)
        temp_avg (float): 
            Average temperature in Celsius
        temp_comp (float): 
            Comparison temperature in Celsius
        meter_corr (float): 
            Average correction for average length of a leveling staff pair meter
        azim_to_moon (geod.Angle):
            Azimuth to the Moon
        zenith_ang_to_moon (geod.Angle):
            Zenithal distance of the Moon
        azim_to_sun (obj.Angle): 
            Azimuth to the Sun
        zenith_ang_to_sun (geod.Angle):
            Zenithal distance of the Sun
        azim_of_line (geod.Angle): 
            Azimuth of the measured line, from starting point
        path_length (float):
            Length of the leveling path in meters

    Returns:
        dh_cor (float):
            Corrected height difference
        cor_temp
            Temperature correction
        cor_comp
            Comparison correction
        cor_lunsol
            Lunisolar correction
    """
    cor_temp = niv_correction_temp(dh, exp, temp_avg, temp_comp)
    cor_comp = niv_correction_comp(dh, meter_corr)
    cor_lunsol = (niv_correction_lunisolar(azim_to_moon, azim_of_line, 
                                    zenith_ang_to_moon, path_length, 'moon')
            + niv_correction_lunisolar(azim_to_sun, azim_of_line, 
                                    zenith_ang_to_sun, path_length, 'sun'))
    dh_cor = dh + cor_comp + cor_lunsol + cor_temp
    return dh_cor, cor_temp, cor_comp, cor_lunsol    
    