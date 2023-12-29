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
        map[-point[1]+center[1],point[0]+center[0]] = 1
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
    fig.set_size_inches((20, 20), forward=False)
    plt.show(block=True)

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
def fromCameraToMap(camera_pose_in_map=[0,0,0.0], points=[],number_of_cells_in_meter=4):
    # Camera pose x and y in map and orientation theta in radians
    processed_points = []
    x_cam = camera_pose_in_map[0]
    y_cam = camera_pose_in_map[1]
    theta = camera_pose_in_map[2]
    processed_points.append([x_cam,y_cam,8])
    cam_pose = np.array([x_cam,y_cam,0])
    # Rotation around y axis
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    K = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]])

    for point in points:
        # point heigh i and width j 
        x = point[1]
        y = point[0]
        # print("Should be in pixels")
        # print([x,y])
        d = point[2]
        normalized_coordinates = np.array([d*x/np.sqrt(x**2 + y**2), d*y/np.sqrt(x**2 + y**2),d])
        # print("Should be in meters")
        # print(normalized_coordinates)
        # Multiply K and R and normalized coordinates
        transformed_coordinates = np.dot(np.dot(K, R), normalized_coordinates)
        transformed_coordinates = [int(x*number_of_cells_in_meter) for x in transformed_coordinates]
        result = transformed_coordinates + cam_pose
        # print("Should be in map coordinates")
        # print(result)
        processed_points.append(result)
        
    return processed_points