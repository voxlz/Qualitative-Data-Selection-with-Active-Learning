from enum import Enum

class Datamix(str, Enum):
    v2          = "v2"       
    hard        = "hard"     
    mixed       = "mixed"    
    noisy       = "noisy"  
    duplicate   = "duplicate"
    imageNet    = "imageNet"
    undefined   = "undefined"
