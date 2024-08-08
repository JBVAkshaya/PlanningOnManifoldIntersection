import numpy as np

def get_state(cell_pose, num_rob, struct_type, min_len=0.5):
    
    if struct_type =='S':
        tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = r1.copy()
        for i in range(1, num_rob):
            tmp = np.array([cell_pose[0], cell_pose[1] + i * min_len, cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
            r1 = np.around(tmp, decimals=2)
            arr = np.append(arr,r1,0)

        return arr
    elif struct_type =='T':
        tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = r1.copy()
        
        tmp = np.array([cell_pose[0], cell_pose[1] + 2.0 * min_len, cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)

        tmp = np.array([cell_pose[0] + min_len, cell_pose[1] + min_len, cell_pose[2],90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)
        return arr
    
    elif struct_type =='I':
        tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = r1.copy()
        
        tmp = np.array([cell_pose[0], cell_pose[1] + 2.0 * min_len, cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)

        tmp = np.array([cell_pose[0] + min_len, cell_pose[1] + min_len, cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)

        tmp = np.array([cell_pose[0] + 2.0 * min_len, cell_pose[1], cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)
        
        tmp = np.array([cell_pose[0] + 2.0 * min_len, cell_pose[1] + 2.0 * min_len, cell_pose[2], 90.0*np.pi/180., 
                45.0*np.pi/180., 45.0*np.pi/180.])                         
        r1 = np.around(tmp, decimals=2)
        arr = np.append(arr,r1,0)
        return arr
