import os
import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt

# Function to read local positions from a text file
def read_local_positions(file_path):
    local_positions = []

    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comments (lines starting with '#')
            if not line.startswith('#'):
                # Split the line into local position values
                local_position = list(map(float, line.split()))
                local_positions.append(local_position)

    return np.array(local_positions, float)

# Function to read 6D poses from a text file
def read_6d_poses(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comments (lines starting with '#')
            if not line.startswith('#'):
                # Split the line into pose values
                pose = list(map(float, line.split()))

    return np.array(pose)

# Combine local positions and 6D poses to calculate global positions
def load_global_position(image_index):
    txt_filename = f"{image_index:06d}.txt"
    object_pose_path = os.path.join("../dataset/box_real/ObjectPose", txt_filename)
    local_position_path = "LocalObjectPoints.txt"

    # Read local positions, 6D poses
    local_positions = read_local_positions(local_position_path)
    object_pose = read_6d_poses(object_pose_path)

    global_positions = []
    x, y, z, rx, ry, rz = object_pose
    for local_position in local_positions:
        rotation_matrix = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
        global_position = np.dot(rotation_matrix, local_position) + np.array([x, y, z])
        global_positions.append(global_position)

    return np.array(global_positions)

def calculate_intrinsic():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:8].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('CalibrationImages/*.jpeg')
    for fname in images:
        print(fname, end = ": ")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print('success')
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 8), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print('fail')
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx)
    print(dist)

# Function to project 3D points to 2D image points
def project_points(object_points):
    # Intrinsic matrix (replace with your actual values)
    """K = np.array([[580.5, 0.0, 320.0],
                  [0.0, 580.5, 240.0],
                  [0.0, 0.0, 1.0]])"""

    """K = np.array([[1305, 0.0, 960.0],
                  [0.0, 1305, 537.0],
                  [0.0, 0.0, 1.0]])"""

    K = np.array([[919.162, 0.0, 956.68],
                  [0.0, 919.322, 551.21],
                  [0.0, 0.0, 1.0]])  # real camera

    # Given rotation angles in degrees
    # rx, ry, rz = np.deg2rad(0), np.deg2rad(215), np.deg2rad(90)  # sim camera 640*480
    rx, ry, rz = np.deg2rad(0), np.deg2rad(225), np.deg2rad(90)  # sim camera 1080p
    # Calculate rotation matrices
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    R = np.dot(Ry, Rz)

    R = np.array([[-0.029, 0.802, -0.596],
                  [1.0, 0.027, -0.012],
                  [0.006, -0.596, -0.803]])  # real camera

    # t = np.array([1.9, 0, 2.3])  # sim camera 640*480
    t = np.array([1.6, 0, 1.0])  # sim camera 1080p
    t = np.array([1.083, -0.008, 0.715])  # real camera
    R = R.T
    t = np.expand_dims(-np.dot(R, t), 1)

    distCoeffs = np.zeros((8, 1), dtype='float32')
    # distCoeffs = np.array([0.556, -2.629, 0., -0., 1.475, 0.433, -2.455, 1.406])
    transformed_points = cv2.projectPoints(object_points, R, t, K, distCoeffs)[0]

    # Convert to numpy array and reshape for convenience
    image_points = transformed_points.reshape(-1, 2)
    x_range = np.ptp(image_points[:, 0])
    y_range = np.ptp(image_points[:, 1])

    return image_points, x_range, y_range

def save_label(image_index, image_points, x_range, y_range):
    txt_filename = f"{image_index:06d}.txt"
    label_path = os.path.join("../dataset/box_real/labels", txt_filename)
    with open(label_path, "w") as txt_file:
        txt_file.write("1 ")
        for point in image_points:
            txt_file.write(f"{point[0] / 1920} {point[1] / 1080} ")

        txt_file.write(f"{x_range / 1920} ")
        txt_file.write(f"{y_range / 1080} ")

def generate_ply_file():
    # Define the corner points of the box
    corner_points = [
        (-0.15, -0.1, 0),
        (-0.15, -0.1, 0.2),
        (-0.15, 0.1, 0),
        (-0.15, 0.1, 0.2),
        (0.15, -0.1, 0),
        (0.15, -0.1, 0.2),
        (0.15, 0.1, 0),
        (0.15, 0.1, 0.2)
    ]

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(corner_points)

    # Save the PointCloud to a PLY file
    o3d.io.write_point_cloud("box.ply", pcd, 'auto', True)

def generate_labels():
    for image_index in range(108):
        # Object points
        object_points = load_global_position(image_index)
        # Project 3D object points to 2D image points
        image_points, x_range, y_range = project_points(object_points)
        # write label file
        save_label(image_index, image_points, x_range, y_range)

        # draw key points on image
        # Load image
        image = cv2.imread(f'../dataset/box_real/JPEGImages/{image_index:06d}.jpeg')
        # Scale and translate the image points to fit into the loaded image
        scale_factor = 1.0
        translation_offset = np.array([0, 0])
        scaled_image_points = (image_points * scale_factor + translation_offset).astype(int)
        # Draw the projected points on the image
        for point in scaled_image_points:
            cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
        image_path = os.path.join("../dataset/box_real/LabeledImages", f"{image_index:06d}.jpeg")
        cv2.imwrite(image_path, image)
        # Show the image with projected points
        # cv2.imshow('Projected Image Points', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def read_transformation_matrices(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Assuming each matrix has 4x4 dimensions
    camera_to_world_matrix = np.array([list(map(float, lines[i+1].split())) for i in range(4)])
    camera_coordinates_matrix = np.array([list(map(float, lines[i+6].split())) for i in range(4)])
    world_coordinates_matrix = np.array([list(map(float, lines[i+11].split())) for i in range(4)])

    return camera_to_world_matrix, camera_coordinates_matrix, world_coordinates_matrix

# Combine local positions and 6D poses to calculate global positions
def load_global_position_from_real_image(image_index, object_pose):
    local_position_path = "LocalObjectPoints.txt"

    # Read local positions, 6D poses
    local_positions = read_local_positions(local_position_path)

    global_positions = []
    x, y, z, rx, ry, rz = object_pose
    for local_position in local_positions:
        rotation_matrix = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
        global_position = np.dot(rotation_matrix, local_position) + np.array([x, y, z])
        global_positions.append(global_position)

    return np.array(global_positions)

def transform_box_world(box_world_matrix):
    # Extract translation components
    translation = box_world_matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = box_world_matrix[:3, :3]

    # Convert rotation matrix to Euler angles (rx, ry, rz)
    rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    ry = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return [translation[0], translation[1], 0, 0, 0, rz-np.pi/2]

def adjust_pose_for_real_images():
    pose_folder = '../dataset/real_box/pose_ground_truth'
    image_folder = '../dataset/real_box/Images'
    jpeg_folder = '../dataset/real_box/JPEGImages'
    pose_files = [file for file in os.listdir(pose_folder) if file.endswith(".txt")]
    image_files = [file for file in os.listdir(image_folder) if file.endswith(".jpeg")]

    image_index = 0
    for pose_filename, image_filename in zip(pose_files, image_files):
        # Read transformation matrices from the pose file
        pose_file_path = os.path.join(pose_folder, pose_filename)
        camera_to_world, box_camera, box_world = read_transformation_matrices(pose_file_path)

        txt_filename = f"{image_index:06d}.txt"
        object_path = os.path.join("../dataset/real_box/ObjectPose_Adjust", txt_filename)
        """box_parameters = transform_box_world(box_world)
        # Write the result to a text file
        with open(object_path, 'w') as output_file:
            output_file.write(" ".join(map(str, box_parameters)))"""

        """if image_index == 107:
            # Read the corresponding image
            object_pose = read_6d_poses(object_path)
            object_points = load_global_position_from_real_image(image_index, object_pose)
            # Project 3D object points to 2D image points
            image_points, x_range, y_range = project_points(object_points)
            image_file_path = os.path.join(image_folder, image_filename)
            # Read and process the image
            image = cv2.imread(image_file_path)
            scale_factor = 1.0
            translation_offset = np.array([0, 0])
            scaled_image_points = (image_points * scale_factor + translation_offset).astype(int)
            # Draw the projected points on the image
            for point in scaled_image_points:
                cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

            # Show the image with projected points
            cv2.imshow('Projected Image Points', cv2.resize(image, (960, 540)))

            # Adjust object pose
            while True:
                key = cv2.waitKey(0)
                if key == ord('w'):
                    object_pose[0] -= 0.0005  # Move forward
                elif key == ord('s'):
                    object_pose[0] += 0.0005  # Move backward
                elif key == ord('a'):
                    object_pose[1] -= 0.0005  # Move left
                elif key == ord('d'):
                    object_pose[1] += 0.0005  # Move right
                elif key == ord('q'):
                    object_pose[5] -= 0.003  # Move left
                elif key == ord('e'):
                    object_pose[5] += 0.003  # Move right
                elif key == ord('c'):
                    print("Quitting...")
                    break
                object_points = load_global_position_from_real_image(image_index, object_pose)
                # Project 3D object points to 2D image points
                image_points, x_range, y_range = project_points(object_points)
                image_file_path = os.path.join(image_folder, image_filename)
                # Read and process the image
                image = cv2.imread(image_file_path)
                scale_factor = 1.0
                translation_offset = np.array([0, 0])
                scaled_image_points = (image_points * scale_factor + translation_offset).astype(int)
                # Draw the projected points on the image
                for point in scaled_image_points:
                    cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

                # Show the image with projected points
                cv2.imshow('Projected Image Points', cv2.resize(image, (960, 540)))

            cv2.destroyAllWindows()
            # Write the result to a text file
            with open(object_path, 'w') as output_file:
                output_file.write(" ".join(map(str, object_pose)))"""

        image_index += 1

if __name__ == "__main__":
    all_x_positions = []
    all_y_positions = []
    all_z_rotations = []
    for image_index in range(108):
        # Object points
        txt_filename = f"{image_index:06d}.txt"
        object_pose_path = os.path.join("../dataset/box_real/ObjectPose", txt_filename)
        object_pose = read_6d_poses(object_pose_path)

        all_x_positions.append(object_pose[0])
        all_y_positions.append(object_pose[1])
        all_z_rotations.append(object_pose[5])

    # Create a heatmap
    heatmap, x_edges, y_edges = np.histogram2d(all_x_positions, all_y_positions, bins=(100, 100))

    # Display the heatmap
    plt.imshow(heatmap.T, extent=(x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Object Pose Density Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

    # Create a histogram for rz values
    plt.hist(all_z_rotations, bins=50, color='blue', edgecolor='black', alpha=0.7)

    # Customize the plot
    plt.title('Distribution of rz Values')
    plt.xlabel('rz Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()