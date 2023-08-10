r1 = {
    
         # x and y coordinates of the bottom-left corner of the rectangle
         'x': 2 , 'y': 4,
         
         # Width and Height of rectangle
         'w':5,'h':12}

def find_rec_overlap(rec1,rec2):
    
    #set output dict full of zeros
    x,y,w,h = 0,0,0,0
    
    # Find if they overlap in x
    # if rec2 is on right and they overlap, ok.  If rec2 is on left and they don't overlap, exit.
    if (rec1['x']+rec1['w']) > rec2['x']:
        x = rec2['x']
        w = rec1['x']-x
    elif (rec2['x']+rec2['w']) < rec1['x']:
        return False
    else:
        x = rec1['x']
        w = rec2['x']-x
     
    # Find if they overlap vertically
    # if rec2 is on above and they overlap, ok.  If rec2 is on under and they don't overlap, exit.
    if (rec1['y']+rec1['h']) > rec2['y']:
        y = rec2['y']
        h = rec1['y']-y
    elif (rec2['y']+rec2['h']) < rec1['y']:
        return False
    else:
        y = rec1['y']
        h = rec2['y']-y
        
    rec_overlap = {'x':x ,'y': y, 'w': w,'h': h}
    
    return rec_overlap

def calc_overlap(coord1, dim1, coord2, dim2):
    """ generalized func to find if there is overlap and return lowest point of overlap and length of overlap."""
    
    # find the largest coord
    coord_max = max(coord1, coord2)
    # find the smallest coord+dim
    coord_dim_min = min(coord1+dim1, coord2+dim2)
    
    #if the largest coord is bigger than smallest coord+dim, they do NOT overlap
    if coord_max >= coord_dim_min:
        return (None,None)
    
    overlap = coord_dim_min - coord_max
    
    return (coord_max, overlap)

def rect_overlap(rec1, rec2):
    
    # See if they overlap in x
    x_overlap, w_overlap = calc_overlap(rec1['x'],rec1['w'],rec2['x'],rec2['w'])
    
    #see if they overlap in y
    y_overlap, h_overlap = calc_overlap(rec1['y'],rec1['h'],rec2['y'],rec2['h'])
    
    # if no overlap at all, ret false
    if w_overlap==h_overlap==None:
        print 'No overlap!'
        return False       
    
    # There is at least a shared edge
    rec_overlap = {'x':x_overlap ,'y': y_overlap, 'w': w_overlap,'h': h_overlap}
    
    return rec_overlap

