import cv2
import os
import numpy as np
import pykitti
import matplotlib.pyplot as plt
import PIL.Image as pil
import random
from projection_helpers import reprojection_pc
from projection_helpers import reprojection_pc_with_pattern
from projection_helpers import image_gradients

import torch
from torchvision import transforms
from skimage.measure import compare_ssim
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace
import networks
from utils import download_model_if_doesnt_exist
from options import MonodepthOptions
from layers import transformation_from_parameters
from datasets import KITTIOdomDataset


def read_base_dir(dir_rec_file):
    with open(dir_rec_file, 'r') as bdf:
        dir_path = bdf.read()
    return dir_path


class DeepDirectVO:
    def __init__(self, _host_frame, _target_frame):
        '''
        initialize the randpattern based photometric residual wrapper
        :param _host_frame: numpy ndarray H x W x 3 image.
        :param _target_frame: numpy ndarray image, same dimension as above.
        '''
        # load options
        options = MonodepthOptions()
        opts = options.parse()
        self.opt = opts
        self.num_input_frames = len(self.opt.frame_ids)
        # init model
        self.model_name = "mono_1024x320"

        download_model_if_doesnt_exist(self.model_name)
        self.encoder_path = os.path.join("models", self.model_name, "encoder.pth")
        self.depth_decoder_path = os.path.join("models", self.model_name, "depth.pth")
        self.pose_encoder_path = os.path.join("models", self.model_name, "pose_encoder.pth")
        self.pose_decoder_path = os.path.join("models", self.model_name, "pose.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.pose_encoder = networks.ResnetEncoder(self.opt.num_layers, False, 2)
        # self.pose_encoder = networks.PoseCNN(self.num_input_frames if self.opt.pose_model_input == "all" else 2)
        self.pose_decoder = networks.PoseDecoder(self.pose_encoder.num_ch_enc, 1, 2)
        # self.pose_decoder = networks.PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1,
        #                                          num_frames_to_predict_for=2)

        self.loaded_dict_enc = torch.load(self.encoder_path, map_location='cpu')
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)

        self.loaded_dict_pose_enc = torch.load(self.pose_encoder_path, map_location='cpu')
        self.filtered_dict_pose_enc = {k: v for k, v in self.loaded_dict_pose_enc.items() if
                                       k in self.pose_encoder.state_dict()}
        self.pose_encoder.load_state_dict(self.filtered_dict_pose_enc)

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.loaded_dict_pose = torch.load(self.pose_decoder_path, map_location='cpu')
        self.pose_decoder.load_state_dict(self.loaded_dict_pose)

        self.encoder.eval()
        self.depth_decoder.eval()

        self.pose_encoder.eval()
        self.pose_decoder.eval()
        self.isgood = []

        # define frames
        self.host_frame = _host_frame
        self.target_frame = _target_frame
        self.host_frame_dx, self.host_frame_dy = image_gradients(self.host_frame)
        self.target_frame_dx, self.target_frame_dy = image_gradients(self.target_frame)

        # dso's pattern:
        self.residual_pattern = np.array([[0, 0],
                                          [-2, 0],
                                          [2, 0],
                                          [-1, -1],
                                          [1, 1],
                                          [-1, 1],
                                          [1, -1],
                                          [0, 2],
                                          [0, -2],
                                          ])

    def gen_pyramid(self, lvl):
        host_pyrd = []
        trgt_pyrd = []
        for ii in range(lvl):
            scale_factor = 2 ** (-ii)
            host_scale_img = cv2.resize(self.host_frame, (0, 0), fx=scale_factor, fy=scale_factor)
            target_scale_img = cv2.resize(self.target_frame, (0, 0), fx=scale_factor, fy=scale_factor)
            host_pyrd.append(host_scale_img)
            trgt_pyrd.append(target_scale_img)
        return host_pyrd, trgt_pyrd

    def gen_pyramid_img(self, img, lvl):
        res_pyrd = []
        for ii in range(lvl):
            scale_factor = 2 ** (-ii)
            res_scale_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            res_pyrd.append(res_scale_img)
        return res_pyrd

    def pose_opts(self):
        # take the depth map of one image and derive the relative pose of two frames.
        pass

    def pose_infer(self, img1, img2):
        feed_height = self.opt.height
        feed_width = self.opt.width
        input_image1_resized = img1.resize((feed_width, feed_height), pil.LANCZOS)
        input_image2_resized = img2.resize((feed_width, feed_height), pil.LANCZOS)
        input_image1_pytorch = transforms.ToTensor()(input_image1_resized).unsqueeze(0)
        input_image2_pytorch = transforms.ToTensor()(input_image2_resized).unsqueeze(0)
        input_images_pytorch = torch.cat([input_image1_pytorch, input_image2_pytorch], 1)
        with torch.no_grad():
            features = self.pose_encoder(input_images_pytorch)
            axisangle, translation = self.pose_decoder([features])
            transf_mat = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()
        return transf_mat

    def depth_infer(self, img):
        # !!!!!!!!!!!!!!!!!!!!! Change scale factor if you use another
        # pretrained model, say, stereo..., they have different basaeline scale...
        SCALE = 0.001
        original_width, original_height = img.size

        feed_height = self.loaded_dict_enc['height']
        feed_width = self.loaded_dict_enc['width']
        input_image_resized = img.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width), mode="bilinear",
                                                       align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        f_cam3 = data.calib.K_cam3[0][0]
        baseline_rgb = data.calib.b_rgb
        # depth is in meter
        depth_img = SCALE * (f_cam3 * baseline_rgb) / disp_resized_np
        return depth_img

    def pselector(self, img, point_n):
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_laplacian = cv2.Laplacian(img, cv2.CV_64F)  # assume the input image is gray scale
        gray_img_laplacian = np.abs(gray_img_laplacian)
        gray_img_laplacian = 1 / (1 + np.exp(-gray_img_laplacian))
        # gray_img_laplacian = gray_img_laplacian / np.max(gray_img_laplacian)
        gheight, gwidth = gray_img_laplacian.shape
        selection_map = np.zeros((gheight, gwidth))
        sel_thresh = 0.98  # 0.18
        sel_m_indx, sel_m_indy = np.where(gray_img_laplacian > sel_thresh)
        rdx = random.sample(range(len(sel_m_indx)), point_n)
        selection_map[sel_m_indx[rdx], sel_m_indy[rdx]] = 1
        return selection_map

    def calRes(self, pose0, pose1, pattern):
        pass

    def plot(self):
        pass

    def reproj_point(self, point, pose1, pose2, K, t_imu2cam):
        '''
            p1 is the 3d point: [x, y, z]
            out put the reprojected point coordinate
            '''
        pxy1 = np.reshape(np.array([point[0], point[1], 1]), (3, 1))
        dep = np.array(point[2])
        kip = np.matmul(np.linalg.inv(K), pxy1)
        dkip = dep * kip
        dkip_homo = np.concatenate((dkip, [[1]]))
        tdkip = np.matmul(np.linalg.inv(t_imu2cam), dkip_homo)
        p1tdkip = np.matmul(pose1, tdkip)
        p2p1tdkip = np.matmul(np.linalg.inv(pose2), p1tdkip)
        tp2p1tdkip = np.matmul(t_imu2cam, p2p1tdkip)
        ktp2p1tdkip = np.matmul(K, tp2p1tdkip[:3])
        reproj = ktp2p1tdkip / ktp2p1tdkip[-1]
        return reproj[:2]

    def photometric_err_rmse(self, p1, p2, pattern, img_w, img_h):
        setting_huberTH = 9
        p1_pattern = p1 + pattern
        p2_pattern = p2 + pattern
        # boundary handle
        p1_pattern[p1_pattern < 0] = 0
        p2_pattern[p2_pattern < 0] = 0
        p1_pattern[p1_pattern > img_h] = img_h - 1
        p2_pattern[p2_pattern > img_w] = img_w - 1
        residual = 0
        for ind in range(len(pattern)):
            color1 = self.host_frame[p1_pattern[ind][1], p1_pattern[ind][0], :]
            color2 = self.target_frame[p2_pattern[ind][1], p2_pattern[ind][0], :]
            tmp_res = np.sum(np.abs(color1 - color2)) / 3
            #         print("tmp_res: {}".format(tmp_res))
            if tmp_res < setting_huberTH:
                hw = 1
            else:
                hw = setting_huberTH / tmp_res
            residual += hw * tmp_res * tmp_res * (2 - hw)
        return residual

    # make a new photometric error and take consideration on the local gradient, ||Ix^2 + Iy^2||
    # set up three configurations, which fit for different gradient levels
    # this requires my experiments on different regions, and need extensive experiments over different images
    # borrow the idea from SIFT, we can do the following extensions:
    # 1. extend the photometric error to consider the orientation
    #       rotate the pattern 36 times, and take the smallest error (can be parallelrized)
    # 2. extend the photometric error to consider the exposure and illumination
    #       apply the exposure affine model like DSO
    # 3. extend the photometric error to consider outliers (occlusion, dynamic models)
    #       RANSAC over candidate points and update the point weight matrix: W in \Delta pose = H^{-1}Wb.
    # 4. extend the photometric error to consider scale changes locally
    #       since we are applying the gradient method which only focus on local region, scale change won't be too sharp
    #       we can introduce a set of scaled pattern to try on.
    # How to model those scaled patterns? One way is to derive by geometry, another way is to derive by the statistic
    # models.

    def photometric_error_p2p(self, p1, p2, img_w, img_h):
        is_good = False
        if p2[0] > (img_h - 1) or p2[0] < 0 or p2[1] > (img_w - 1) or p2[1] < 0:
            return is_good, 0
        else:
            setting_huberTH = 9
            color1 = self.host_frame[p1[0], p1[1]]
            color2 = self.target_frame[p2[0], p2[1]]
            tmp_res = np.fabs(int(color1) - int(color2))
            if tmp_res < setting_huberTH:
                hw = 1
            else:
                hw = setting_huberTH / tmp_res
            residual = hw * tmp_res * tmp_res * (2 - hw)
            is_good = True
            return is_good, residual

    def photometric_err_weighted(self, p1, p2, pattern, img_w, img_h):
        setting_huberTH = 9
        p1_pattern = p1 + pattern
        p2_pattern = p2 + pattern
        # boundary handle
        p1_pattern[p1_pattern < 0] = 0
        p2_pattern[p2_pattern < 0] = 0
        p1_pattern[p1_pattern > img_h] = img_h - 1
        p2_pattern[p2_pattern > img_w] = img_w - 1
        residual = 0
        # how about do the small rotation and take the min?
        for ind in range(len(pattern)):
            dist_weight = 1 / np.linalg.norm(p1_pattern[ind])
            color1 = self.host_frame[p1_pattern[ind][1], p1_pattern[ind][0], :]
            color2 = self.target_frame[p2_pattern[ind][1], p2_pattern[ind][0], :]
            tmp_res = np.sum(np.abs(color1 - color2)) / 3
            #         print("tmp_res: {}".format(tmp_res))
            if tmp_res < setting_huberTH:
                hw = 1
            else:
                hw = setting_huberTH / tmp_res
            residual += dist_weight * hw * tmp_res * tmp_res * (2 - hw)
        return residual

    def getInterpolated33(self, pf, pi):
        '''
        make sure that the pf and pi are not OOB
        '''
        dx = pf[0] - pi[0] # height
        dy = pf[1] - pi[1] # width
        dxdy = dx * dy




    def getInterpolated31(self, pf, pi):
        '''
        get the interpolated reprojected pixel value
        '''
        pass


    def calcResAndGS_vectorized(self, reproj_ppair_list, img_w, img_h):
        '''
        calculate the reprojection error for all selected points
        '''
        setting_huberTH = 9
        if not type(reproj_ppair_list) == np.ndarray:
            reproj_ppair_list = np.array(reproj_ppair_list)
        reproj_ppair_list = np.transpose(reproj_ppair_list, (2, 0, 1)) # 1300, 9, 4
        reproj_ppair_list2 = np.reshape(reproj_ppair_list, (-1, 4))
        # noob means not out of boundary
        reproj_noob = np.logical_and(
            np.logical_and(reproj_ppair_list2[:, 2] > 0, reproj_ppair_list2[:, 2] < gheight - 1),
            np.logical_and(reproj_ppair_list2[:, 3] > 0, reproj_ppair_list2[:, 3] < gwidth - 1))
        reproj_noob = np.reshape(reproj_noob, (-1, 9))
        reproj_noob = np.all(reproj_noob, axis=1)
        self.isgood = reproj_noob
        good_p = reproj_ppair_list[np.where(reproj_noob)]
        good_p = np.reshape(good_p, (-1, 4)) # height, width, height, width
        good_p1 = good_p[:, :2].astype(int)
        good_p2 = good_p[:, 2:].astype(int)
        good_p2_f = good_p[:, 2:]
        # get interpolate31
        # dx_dy = good_p2_f - good_p2
        rlR = self.getInterpolated31(good_p2_f, good_p2)
        color1 = self.host_frame[good_p1[:, 0], good_p1[:, 1]]
        color2 = self.target_frame[good_p2[:, 0], good_p2[:, 1]]
        color1 = np.array(color1).astype(int)
        color2 = np.array(color2).astype(int)
        residual = np.abs(color1 - color2).astype(float)
        # huber norm
        huber_in_ind = np.where(residual <= setting_huberTH)
        huber_out_ind = np.where(residual > setting_huberTH)
        residual[huber_in_ind] = 0.5 * residual[huber_in_ind] * residual[huber_in_ind]
        residual[huber_out_ind] = setting_huberTH * (residual[huber_out_ind] - 0.5 * setting_huberTH)
        return np.sum(residual)

    def calcResAndGS_with_pattern(self, reproj_ppair_list, img_w, img_h):
        if not type(reproj_ppair_list) == np.ndarray:
            reproj_ppair_list = np.array(reproj_ppair_list)
        reproj_ppair_list = np.transpose(reproj_ppair_list, (2, 0, 1))
        residuals = 0
        for reproj_pair in reproj_ppair_list:
            for p2p in reproj_pair:
                p1 = p2p[:2].astype(int)
                p2 = p2p[2:].astype(int)
                is_good, residual = self.photometric_error_p2p(p1, p2, img_w, img_h)
                if not is_good:
                    break
                residuals += residual
        return residuals

    def calcResAndGS(self, reproj_points, img_w, img_h):
        residuals = 0
        for p2p in reproj_points.T:
            p1 = p2p[:2].astype(int)
            p2 = p2p[2:].astype(int)
            residuals += self.photometric_err_weighted(p1, p2, self.residual_pattern, img_w, img_h)
        return residuals

    def patch_loss(self, point1, point2, patchsize, rpattern, w, h):
        x = range(-patchsize, patchsize)
        y = range(-patchsize, patchsize)
        xx, yy = np.meshgrid(x, y)
        xy2_patch = np.dstack((point2[0] + xx, point2[1] + yy))
        xy2_patch_reshaped = np.reshape(xy2_patch, (-1, 2))
        pe_list = []
        for xy2_p in xy2_patch_reshaped:
            #         tmp_err = photometric_err_rmse(xy, xy2_p, rpattern, w, h)
            tmp_err = self.photometric_err_weighted(point1, xy2_p, rpattern, w, h)
            pe_list.append(tmp_err)
        pe_list = np.array(pe_list) / len(rpattern)
        pe_list = np.reshape(pe_list, (2 * patchsize, 2 * patchsize))
        return pe_list

    def pselect_pyramid(self, lvl):
        host_pyramid, target_pyramid = self.gen_pyramid(lvl)  # 3 lvls
        host_sel_pyramid = [self.pselector(host_scale_img) for host_scale_img in host_pyramid]
        target_sel_pyramid = [self.pselector(target_scale_img) for target_scale_img in target_pyramid]
        return host_sel_pyramid, target_sel_pyramid

    def depth_sel_pyramid(self, depth_img, selection_map, lvl):
        '''
        select host depth image
        '''
        host_pyramid, target_pyramid = self.gen_pyramid_img(depth_img, lvl)  # 3 lvls
        host_sel_pyramid = [self.pselector(host_scale_img) for host_scale_img in host_pyramid]
        target_sel_pyramid = [self.pselector(target_scale_img) for target_scale_img in target_pyramid]
        return host_sel_pyramid, target_sel_pyramid


if __name__ == "__main__":
    # some test variables
    basedir = read_base_dir("./dataset_base_dir")
    odometry_dir = os.path.join(basedir, "data_odometry_color/dataset/")
    # basedir = '/home/ran/Documents/Dataset/KITTI'
    # basedir = '/mnt/sda1/Dataset/Kitti/raw_data/'
    date = '2011_09_26'
    drive = '0005'
    data = pykitti.raw(basedir, date, drive, dataset="sync")
    img_idx = 98
    host_frame = cv2.imread(data.cam3_files[img_idx], 0)
    target_frame = cv2.imread(data.cam3_files[img_idx + 1], 0)
    ddvo = DeepDirectVO(host_frame, target_frame)
    depth_img = ddvo.depth_infer(data.get_cam3(img_idx))
    trans_mat = ddvo.pose_infer(data.get_cam3(img_idx), data.get_cam3(img_idx + 1))
    print('trans_mat: {}'.format(trans_mat))
    print('end')
    # note that the T_w_imu is transforming from imu coordinate into world coordinate
    pose1 = data.oxts[img_idx].T_w_imu
    pose2 = data.oxts[img_idx + 1].T_w_imu
    Timu = data.calib.T_cam3_imu  # imu to camera
    Tcam = np.linalg.inv(Timu)  # camera to imu
    t1 = np.matmul(pose1, Tcam)
    t2 = np.matmul(np.linalg.inv(pose2), t1)
    t3 = np.matmul(Timu, t2)
    selection_map = ddvo.pselector(ddvo.host_frame, 13000)
    non_zero_indice = np.where(selection_map > 0)
    gheight, gwidth = selection_map.shape
    selection_depth = np.zeros((gheight, gwidth))
    selection_depth[non_zero_indice[0], non_zero_indice[1]] = depth_img[non_zero_indice[0], non_zero_indice[1]]
    # print(selection_map)
    # reproj_ppairs = reprojection_pc(selection_depth, pose1, pose2, Timu, data.calib.K_cam3) # 4 by n matrix
    # energy = rdpattern.calcResAndGS(reproj_ppairs, gwidth, gheight)
    # print(reproj_ppairs)

    reproj_ppair_list, pc_frame1_camera, T_ref2New = reprojection_pc_with_pattern(selection_map,
                                                                                  ddvo.residual_pattern, pose1, pose2, Timu, data.calib.K_cam3)
    r = ddvo.calcResAndGS_vectorized(reproj_ppair_list, gwidth, gheight)
    print('hello', r)

    # this pose estimator is off, since the pretrained network is in odometry data, doesn't support general pose estimation
    # try estimate pose using the iterative method and optimizer.
    # 1. select pixels
    # 2. estimate coarse depth
    # 3. reprojection and collect energy
    # 4. estimate pose using GN method: Hx = b, inc = inv(H)*b, ref2New = inc[:6]
    # instead of trying to predict pose directly from the RGB image, why can't we train it on depth image?
    # depth image has less dimensions, and are easier for capture the geometrical relationships