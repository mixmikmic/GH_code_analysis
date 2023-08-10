get_ipython().run_cell_magic('file', 'movement_generation_training.py', "\n# Generates the movements according to:\n# Flash, Tamar and Neville Hogan. 1985. The Coordination of Arm Movements: An Experimentally Confirmed \n# Mathematical Model. The Journal of Neuroscience 5 (7): 1688-1703\ndef movement_generation_training(xstart,xdest,MT,t):\n    '''\n    xstart,ystart: initial position of the trajectory\n    MT: total time spent doing the trajectory\n    t: current time\n    \n    returns a matrix: [[x0,y0],[x1,y1],...]\n    '''\n    x_t=xstart+(xstart-xdest)*(15*(t/MT)**4-6*(t/MT)**5-10*(t/MT)**3)\n    return numpy.array(x_t)")

get_ipython().run_cell_magic('file', 'rot_array.py', '\ndef rot_array(a,r):\n    return numpy.round(numpy.array([a[0]*numpy.cos(r)-a[1]*numpy.sin(r),a[0]*numpy.sin(r)+a[1]*numpy.cos(r)]),2)')

get_ipython().run_cell_magic('file', 'vrep_training.py', '\nimport os\n\np = None\n\n# To automate the process is necessary to avoid the firewall message by deactivating it!!!!\n\n\np = subprocess.Popen([vrep_location, \'-h\', \'-s\', \'-q\', \\\n                      os.getcwd()+\'/VREP_scenes/Baxter_IK_felt_pen_pick-and-place_learning_IJCNN2016.ttt\'])  \n\ntime.sleep(1.0)\n\n# Object names (used inside the simulation)\n# They are used to retrieve the object handles\ns0_name = \'Baxter_leftArm_joint1\'\ns1_name = \'Baxter_leftArm_joint2\'\ne1_name = \'Baxter_leftArm_joint4\'\nw1_name = \'Baxter_leftArm_joint6\'\nXY_pos = \'IK_XY_MASTER\' # Controls the cartesian X,Y values (pen)\nZ_pos = \'IK_Z_MASTER\' # Controls the cartesian Z value (pen)\n\n# simxPauseCommunication(clientID,1);\n# simxSetJointPosition(clientID,joint1Handle,joint1Value,simx_opmode_oneshot);\n# simxSetJointPosition(clientID,joint2Handle,joint2Value,simx_opmode_oneshot);\n# simxSetJointPosition(clientID,joint3Handle,joint3Value,simx_opmode_oneshot);\n# simxPauseCommunication(clientID,0);\n# Above\'s 3 joints will be received and set on the V-REP side at the same time\n\nprint \'Program started\'\nvrep.simxFinish(-1) # just in case, close all opened connections\n\n# Connects to the simulator\nclientID=vrep.simxStart(\'127.0.0.1\',19999,True,True,5000,5)\n\nif clientID!=-1:\n    print \'Connected to remote API server\'\n    res,objs=vrep.simxGetObjects(clientID,vrep.sim_handle_all,vrep.simx_opmode_oneshot_wait) # gets ALL object handles\n    if res==vrep.simx_return_ok:\n        print \'Number of objects in the scene: \',len(objs)\n        res0,XY=vrep.simxGetObjectHandle(clientID,XY_pos,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the IK_XY_MASTER\n        res1,Z=vrep.simxGetObjectHandle(clientID,Z_pos,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the IK_Z_MASTER\n        \n        res2,s0=vrep.simxGetObjectHandle(clientID,s0_name,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the s0 joint\n        res3,s1=vrep.simxGetObjectHandle(clientID,s1_name,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the s1 joint\n        res4,e1=vrep.simxGetObjectHandle(clientID,e1_name,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the e1 joint\n        res5,w1=vrep.simxGetObjectHandle(clientID,w1_name,vrep.simx_opmode_oneshot_wait) # gets specifically the handle for the w1 joint\n\n        if (res1*res2*res3*res4*res5)==vrep.simx_return_ok:\n            print "Ok, I\'m in!"\n\n            joint_list = [s0,s1,e1,w1]\n            joint_positions = []\n\n            #\n            # These are the equivalent to X,Y,Z = 0,0,0 in my system:\n            #\n            \n            # Reads the current XY_Master position [X,Y]\n            print "Reads the current XY_Master position [X,Y]"\n            res,posXY=vrep.simxGetObjectPosition(clientID,XY,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)\n            time.sleep(0.5)\n            \n            # Reads the current Z_Master position [Z]\n            print "Reads the current Z_Master position [Z]"\n            res,posZ=vrep.simxGetObjectPosition(clientID,Z,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)            \n            time.sleep(0.5)\n            \n            print "Initial XY Master Position", posXY\n            print "Initial Z Master Position", posZ\n\n\n            # Lifts the pen to position it in the trajectory\'s starting point\n            # This is acomplished by increasing the Z value (posZ[2])\n            print "Lifts the pen to position it in the trajectory\'s starting point"\n            # I need a smooth movements, otherwise the pen touches the table by mistake.\n            for k in range(2):\n                res = vrep.simxSetObjectPosition(clientID,Z,vrep.sim_handle_parent,[posZ[0], posZ[1], posZ[2]+0.001*(k+1)],vrep.simx_opmode_oneshot_wait)\n                time.sleep(0.5)\n                        \n            # Moves to the first position\n            # This movement is relative to the current XY position!!!\n            print "Moves to the first position"\n            res = vrep.simxSetObjectPosition(clientID,XY,vrep.sim_handle_parent,[posXY[0]+XY_movement[0,0], posXY[1]+XY_movement[0,1], posXY[2]],vrep.simx_opmode_oneshot_wait)\n            time.sleep(0.5)\n            \n            \n            # Puts down the pen\n            print "Puts down the pen"\n            # I need smooth movements, otherwise the pen touches the table by mistake.\n            # Reads the current Z_Master position [Z]\n            res,posZcurrent=vrep.simxGetObjectPosition(clientID,Z,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)            \n            time.sleep(0.5)\n            for k in range(2):\n                res = vrep.simxSetObjectPosition(clientID,Z,vrep.sim_handle_parent,[posZcurrent[0], posZcurrent[1], posZcurrent[2]-0.001*(k+1)],vrep.simx_opmode_oneshot_wait)\n                time.sleep(0.5)            \n            \n            if res!=0:\n                vrep.simxFinish(clientID)\n                print \'Remote API function call returned with error code (start-up): \',res\n          \n            # Reads the current XY_Master position [X,Y]\n            print "Reads the current XY_Master position [X,Y]"\n            res,posXYcurrent=vrep.simxGetObjectPosition(clientID,XY,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)\n            time.sleep(0.5)\n            \n            # Reads the current Z_Master position [Z]\n            print "Reads the current Z_Master position [Z]"\n            res,posZcurrent=vrep.simxGetObjectPosition(clientID,Z,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)            \n            time.sleep(0.5)            \n\n            print "XY Master Position:", posXYcurrent\n            print "Z Master Position:", posZcurrent\n            \n            \n            i = 0\n            for hi in XY_movement:\n                # Reads and saves the current joint positions\n                temp = []\n                for ji in joint_list:\n                    res,joint_pos=vrep.simxGetJointPosition(clientID,ji,vrep.simx_opmode_oneshot_wait)\n                    temp.append(joint_pos)\n                    time.sleep(0.0025)\n                joint_positions.append(temp)\n                \n                if i==0:\n                    if save2file:\n                        numpy.save(base_dir+"/"+sim_set+"/starting_joint_pos.npy",numpy.array(temp))                \n\n                cmd_pos = numpy.array(posXY)+numpy.concatenate([hi,[0]]) # Sums X and Y in the pos and [hi[0],hi[1],0] arrays\n\n                i+=1\n                # Sets the new position\n                res = vrep.simxSetObjectPosition(clientID,XY,vrep.sim_handle_parent,cmd_pos,vrep.simx_opmode_oneshot_wait)\n                time.sleep(0.0025) # 0.05 here was generating too many time-out errors!                \n                \n                if res!=0:\n                    vrep.simxFinish(clientID)\n                    print \'Remote API function call returned with error code (main loop): \',res\n                    break\n            \n\n            # Lifts the pen to position it in the trajectory\'s starting point\n            # I need a smooth movements, otherwise the pen touches the table by mistake.\n            # Reads the current Z_Master position [Z]\n            res,posZcurrent=vrep.simxGetObjectPosition(clientID,Z,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)            \n            time.sleep(0.5)            \n        \n            for k in range(2):\n                res = vrep.simxSetObjectPosition(clientID,Z,vrep.sim_handle_parent,[posZcurrent[0], posZcurrent[1], posZcurrent[2]+0.001*(k+1)],vrep.simx_opmode_oneshot_wait)\n                time.sleep(0.5)            \n\n            \n            # Moves it back to the first position\n            res = vrep.simxSetObjectPosition(clientID,XY,vrep.sim_handle_parent,posXY,vrep.simx_opmode_oneshot_wait)\n            time.sleep(0.5)\n            \n            # Puts down the pen\n            # I need a smooth movements, otherwise the pen touches the table by mistake.\n            # Reads the current Z_Master position [Z]\n            res,posZcurrent=vrep.simxGetObjectPosition(clientID,Z,vrep.sim_handle_parent,vrep.simx_opmode_oneshot_wait)            \n            time.sleep(0.5)\n            for k in range(2):\n                res = vrep.simxSetObjectPosition(clientID,Z,vrep.sim_handle_parent,[posZcurrent[0], posZcurrent[1], posZcurrent[2]-0.001*(k+1)],vrep.simx_opmode_oneshot_wait)\n                time.sleep(0.5)      \n                \n            if res!=0:\n                vrep.simxFinish(clientID)\n                print \'Remote API function call returned with error code (last position): \',res\n                \n        else:\n            print \'Remote API function call returned with error code (object handles): \',res\n    else:\n        print \'Remote API function call returned with error code (first connection): \',res\n#     returncode=vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)\n    vrep.simxFinish(clientID)\nelse:\n    print \'Failed connecting to remote API server\'\nprint \'Program ended\'\n\nif p:\n    # Terminates the process, in the case the connection above failed.\n    p.terminate()')


