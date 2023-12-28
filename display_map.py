import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def DisplayMap(rows,cols,points):
    cmap = colors.ListedColormap(['white','grey','black'])
    bounds = [0, 0.25,0.75, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    map = np.zeros((cols,rows, 1))
    center = [int(rows/2),int(cols/2)]
    for point in points:
        map[-point[1]+center[1],point[0]+center[0]] = point[2]
    fig, ax = plt.subplots()
    ax.imshow(map, cmap=cmap, norm=norm)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    x_tick_labels = [i if i % 10 == 0 else '' for i in range(1,rows+1)]
    y_tick_labels = [i if i % 10 == 0 else '' for i in range(1,cols+1)]
    ax.set_xticks(np.arange(0.5, rows, 1),x_tick_labels)
    ax.set_yticks(np.arange(0.5, cols, 1),y_tick_labels)
    plt.tick_params(axis='both', which='both', bottom=False,   
                    left=False, labelbottom=True, labelleft=True) 
    fig.set_size_inches((8.5, 11), forward=False)
    plt.show()

def fromCameraCoordinatesToMapCoordinates(x,depth,rows,cols):
    depth= 2.5
    number_of_cells_in_meter = 4
    camera_pose_on_map = [0,0]
    meters_per_pixel = 3.0/160
    x = int((x*meters_per_pixel)*number_of_cells_in_meter) + camera_pose_on_map[0]
    print(x)
    alpha = np.cosh(x*meters_per_pixel/depth)
    y = int(depth*np.sin(alpha)*number_of_cells_in_meter) + camera_pose_on_map[1]
    print(y)
    return [x,y,1]

rows= 50
cols = 50
DisplayMap(rows,cols,[[0,0,1],fromCameraCoordinatesToMapCoordinates(-50,2.5,rows,cols)])
