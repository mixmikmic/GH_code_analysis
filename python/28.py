def spiral(n):
    tr = 0
    tl = 0
    bl = 0
    br = 0
    
    while n > 1:
    
        _tr = n**2
        _tl = _tr - (n-1)
        _bl = _tl - (n-1)
        _br = _bl - (n-1)
        
        tr += _tr
        tl += _tl
        bl += _bl
        br += _br
        
        n -= 2
        
    return tr + tl + bl + br + 1

