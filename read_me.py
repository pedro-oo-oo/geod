# -*- coding: utf-8 -*-

def bobux_generator_yada_yada():
    print('haha the real deal is outdated, just read documentation lol')
    
def the_real_deal():
    print("""
Created on Thu Feb 13 16:48:41 2025

@author: pnapi
v.1.0

===============================================================================
===============================================================================

Constants, Errors, Classes and Methods 
    defined in geod.py:

===============================================================================

Constants:
    
        #GRS80 ellipsoid:
    a = 6378137. #bigger semi-axis of grs80 [meters]
    e2 = 0.00669438002290 #eccentricity of grs80 squared 

        #Krasowski ellipsoid:
    akr = 6378245. #bigger semi-axis of Krasowski ellipsoid [meters]
    e2kr = 0.00669342162296 #eccentricity of Krasowski squared 

    rho = 206265.0 #rho
    ro = rho #just to avoid stupidity

        #Projection scales
    m0_1992 = 0.9993 #projection scale PL-1992 <-> GK
    m0_2000 = 0.999923 #projection scale PL-2000 <-> GK
    
===============================================================================

Errors:

    StupidError
        Raised when you commit a felony.
        
    CoordinatesTypeError
        Raised when trying to perform an operation on an invalid coordinate 
        system for that operation
        
    MissingArgError
        Raised when an argument is missing
        
===============================================================================

Classes:
    
    Angle
        Allows to easily format and print a value of an angle, 
        also allows for easy conversions, e.g. grad to degrees,
        as well as some additional functionalities.
        
    Coordinates2d and Coordinates3d
        Allows for quick conversions between coordinate systems
        and is a required type() for most coordinates-related functions to run.
        
===============================================================================

Methods:
(note that this only contains method names, for more info call help(func_name),
don't call me lazy, you wouldn't do this too, I bet)

    info
    approx
    line_length
    find_N
    find_M
    find_R
    ang_red
    convergence
    azim_red
    find_azim
    get_neu_delta_vect
    get_obs_data_from_neu
    orthodromic
    get_orthodromic_end
    lambda_0
    line_red_ell_to_GK
    line_len_ell_to_GK
    line_len_parallel_projection
    neu2XYZ
    projection_scale_GK
    transMatrixA6
    transMatrixA7
    transMatrixA9
    transMatrixL
    
        
""")