import numpy as np


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


def point_cloud(depth, K):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, -1]
    cy = K[1, -1]
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth, -1)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))


def reprojection_pc(sparse_depth, pose0, pose1, T_imu2cam, K_cam):
    '''
    depth_map: depth map prediction
    pose0: frame0 imu to world Transformation matrix
    pose1: frame1 imu to world Transformation matrix
    T_cam2imu: camera to IMU transformation matrix
    K_cam: camera intrinsic
    '''
    # convert sparse depth to point cloud
    # depth estimation is off by 10 times in monodepth2, due to their regularization tricks.
    sparse_pc = point_cloud(sparse_depth * 10, K_cam)
    idx, idy = np.where(sparse_depth > 0)
    sparse_pc_res = sparse_pc[np.where(sparse_depth > 0)]
    num_sparse_pc = sparse_pc_res.shape[0]
    sparse_pc_homo = np.column_stack((sparse_pc_res, np.ones(num_sparse_pc)))
    # camera coordinate to imu coordinate
    pc0_homo_imu = np.dot(np.linalg.inv(T_imu2cam), sparse_pc_homo.T)
    # project imu coordinate into world coordinate
    pc_w_homo = np.dot(pose0, pc0_homo_imu)
    pc_frame1_homo = np.dot(np.linalg.inv(pose1), pc_w_homo)
    # project into second frame's camera coordinate
    pc_frame1_camera = np.dot(T_imu2cam, pc_frame1_homo)
    # convert into pixel coordinate with camera intrinsics.
    pc_frame1_pixel = np.dot(K_cam, pc_frame1_camera[:3, :])
    pc_frame1_pixel = pc_frame1_pixel[:2, :] / pc_frame1_pixel[2, :]
    # the pixel's index is the order in the original image
    #     pc_frame1_pixel = np.reshape(pc_frame1_pixel, (-1, height, width))
    return np.vstack((np.vstack((idx, idy)),
                      pc_frame1_pixel))  # 4 by n matrix, original pixel coordinate paired with target pixel coordinates


def reprojection_pc_with_pattern(sparse_depth, pattern, pose0, pose1, T_imu2cam, K_cam):
    '''
    depth_map: depth map prediction
    pose0: frame0 imu to world Transformation matrix
    pose1: frame1 imu to world Transformation matrix
    T_cam2imu: camera to IMU transformation matrix
    K_cam: camera intrinsic
    '''
    # convert sparse depth to point cloud
    # depth estimation is off by 10 times in monodepth2, due to their regularization tricks.
    # pattern is nx2 vector, 2 is the x and y coordinate offsets, n is the residual dimension number
    idx, idy = np.where(sparse_depth > 0) # idx is the row coordinate, idy is the column coordinate
    reproj_pc_list = []
    for _, offset in enumerate(pattern):
        T_ref2New = np.matmul(pose0, np.linalg.inv(T_imu2cam))
        T_ref2New = np.matmul(np.linalg.inv(pose1), T_ref2New)
        T_ref2New = np.matmul(T_imu2cam, T_ref2New)
        sparse_depth_shifted = shift_image(sparse_depth, offset[0], offset[1])
        sparse_pc = point_cloud(sparse_depth_shifted * 10, K_cam)
        # idx, idy = np.where(sparse_depth > 0)
        sparse_pc_res = sparse_pc[np.where(sparse_depth > 0)]
        num_sparse_pc = sparse_pc_res.shape[0]
        sparse_pc_homo = np.column_stack((sparse_pc_res, np.ones(num_sparse_pc)))
        # camera coordinate to imu coordinate
        pc0_homo_imu = np.dot(np.linalg.inv(T_imu2cam), sparse_pc_homo.T)
        # project imu coordinate into world coordinate
        pc_w_homo = np.dot(pose0, pc0_homo_imu)
        pc_frame1_homo = np.dot(np.linalg.inv(pose1), pc_w_homo)
        # project into second frame's camera coordinate
        pc_frame1_camera = np.dot(T_imu2cam, pc_frame1_homo)
        # convert into pixel coordinate with camera intrinsics.
        pc_frame1_pixel = np.dot(K_cam, pc_frame1_camera[:3, :])
        pc_frame1_pixel = pc_frame1_pixel[:2, :] / pc_frame1_pixel[2, :]
        pc_frame1_pixel[[0, 1], :] = pc_frame1_pixel[[1, 0], :]
        # the pixel's index is the order in the original image
        # the order is as following: row_origin, col_corigin, row_target, col_target
        reproj_pc_list.append(np.vstack((np.vstack((idx, idy)),
                      pc_frame1_pixel)))  # 4 by n matrix, original pixel coordinate paired with target pixel coordinates
    return reproj_pc_list, pc_frame1_camera, T_ref2New

def shift_image(X, dx, dy):
    '''
     use roll function to circular shift x and y and then zerofill the offset
    :param dx: shift step in x direction, horizental
    :param dy: shift step in y direction, vertical
    :return: shifted matrix
    '''
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image_gradients(img_data):
    # gray_img_data = rgb2gray(img_data)
    gray_img_data = img_data
    dIp_x_plus = shift_image(gray_img_data[:, :], 1, 0)
    dIp_x_minus = shift_image(gray_img_data[:, :], -1, 0)
    dIp_y_plus = shift_image(gray_img_data[:, :], 0, 1)
    dIp_y_minus = shift_image(gray_img_data[:, :], 0, -1)
    dx = 0.5*(dIp_x_plus - dIp_x_minus)
    dy = 0.5*(dIp_y_plus - dIp_y_minus)
    return dx, dy