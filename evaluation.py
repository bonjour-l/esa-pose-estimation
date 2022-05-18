import os
from datetime import time

import cv2
import numpy as np
from plyfile import PlyData
import scipy

from pnp import pnp
import matplotlib.pyplot as plt
from lib.utils.extend_utils.extend_utils import find_nearest_point_idx


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LineModModelDB(object):
    '''
    LineModModelDB is used for managing the mesh of each model
    '''
    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {'8': {}, '4': {}, '12': {}, '16': {}, '20': {}}
    sift_3d={}
    small_bbox_corners={}
    #class_type='cat'
    def __init__(self,name):
        self.name=name
        self.ply_pattern = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/'+name+'/'+name+'.ply'
        self.diameter_pattern = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD_ORIG/'+name+'/distance.txt'
        self.farthest_pattern = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/'+name+'/farthest.txt'
        self.sift_pattern = '/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/'+name+'/'+name+'.txt'


        #self.ply_pattern = os.path.join('/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/', 'cat/cat.ply')
        #self.diameter_pattern = os.path.join('/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD_ORIG/','cat/distance.txt')
        #self.farthest_pattern = os.path.join('/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/','cat/farthest.txt')
        #self.sift_pattern = os.path.join('/home/lin/Documents/6D/pvnet/pvnet-master/data/LINEMOD/','cat/cat.txt')

    def get_corners_3d(self, class_type):
        if class_type in self.corners_3d:
            return self.corners_3d[class_type]

        #corner_pth=os.path.join('/home/lin/Documents/6D/linemod/LINEMOD/', 'cat/corners.txt')
        corner_pth='/media/zhaobotong/ab9dc7e7-9ac1-4a99-aa64-e02a484c8cad/home/lin/Documents/6D/linemod/LINEMOD/'+self.name+'/corners.txt'
        if os.path.exists(corner_pth):
            self.corners_3d[class_type]=np.loadtxt(corner_pth)
            return self.corners_3d[class_type]

        ply_path = self.ply_pattern.format(class_type, class_type)
        ply = PlyData.read(ply_path)
        data = ply.elements[0].data

        x = data['x']
        min_x, max_x = np.min(x), np.max(x)
        y = data['y']
        min_y, max_y = np.min(y), np.max(y)
        z = data['z']
        min_z, max_z = np.min(z), np.max(z)
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        self.corners_3d[class_type] = corners_3d
        np.savetxt(corner_pth,corners_3d)

        return corners_3d

    def get_small_bbox(self, class_type):
        if class_type in self.small_bbox_corners:
            return self.small_bbox_corners[class_type]

        corners=self.get_corners_3d(class_type)
        center=np.mean(corners,0)
        small_bbox_corners=(corners-center[None,:])*2.0/3.0+center[None,:]
        #small_bbox_corners=(corners-center[None,:])+center[None,:]
        self.small_bbox_corners[class_type]=small_bbox_corners

        return small_bbox_corners

    def get_ply_model(self, class_type):
        if class_type in self.models:
            return self.models[class_type]

        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        self.models[class_type] = model
        return model

    def get_diameter(self, class_type):
        if class_type in self.diameters:
            return self.diameters[class_type]

        diameter_path = self.diameter_pattern.format(class_type)
        diameter = np.loadtxt(diameter_path) / 100.
        self.diameters[class_type] = diameter
        return diameter

    def get_centers_3d(self, class_type):
        if class_type in self.centers_3d:
            return self.centers_3d[class_type]

        c3d=self.get_corners_3d(class_type)
        self.centers_3d[class_type]=(np.max(c3d,0)+np.min(c3d,0))/2
        return self.centers_3d[class_type]

    def get_farthest_3d(self,class_type,num=8):
        if class_type in self.farthest_3d['{}'.format(num)]:
            return self.farthest_3d['{}'.format(num)][class_type]

        if num==8:
            farthest_path = self.farthest_pattern.format(class_type,'')
        else:
            farthest_path = self.farthest_pattern.format(class_type,num)
        farthest_pts = np.loadtxt(farthest_path)
        self.farthest_3d['{}'.format(num)][class_type] = farthest_pts
        return farthest_pts

    def get_sift_3d(self,class_type):


        sift_path = self.sift_pattern.format(class_type,'')

        sift_pts = np.loadtxt(sift_path)
        self.sift_3d[class_type] = sift_pts
        return sift_pts

    def get_ply_mesh(self,class_type):
        ply = PlyData.read(self.ply_pattern.format(class_type, class_type))
        vert = np.asarray([ply['vertex'].data['x'],ply['vertex'].data['y'],ply['vertex'].data['z']]).transpose()
        vert_id = [id for id in ply['face'].data['vertex_indices']]
        vert_id = np.asarray(vert_id,np.int64)

        return vert, vert_id

def find_nearest_point_distance(pts1,pts2):
    '''

    :param pts1:  pn1,2 or 3
    :param pts2:  pn2,2 or 3
    :return:
    '''
    idxs=find_nearest_point_idx(pts1,pts2)
    return np.linalg.norm(pts1[idxs]-pts2,2,1)

class Projector(object):
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]]),
        'pascal': np.asarray([[-3000.0, 0.0, 0.0],
                              [0.0, 3000.0, 0.0],
                              [0.0,    0.0, 1.0]])
    }

    def project(self,pts_3d,RT,K_type):
        pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        pts_2d=np.matmul(pts_2d,self.intrinsic_matrix[K_type].T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d

    def project_h(self,pts_3dh,RT,K_type):
        '''

        :param pts_3dh: [n,4]
        :param RT:      [3,4]
        :param K_type:
        :return: [n,3]
        '''
        K=self.intrinsic_matrix[K_type]
        return np.matmul(np.matmul(pts_3dh,RT.transpose()),K.transpose())

    def project_pascal(self,pts_3d,RT,principle):
        '''

        :param pts_3d:    [n,3]
        :param principle: [2,2]
        :return:
        '''
        K=self.intrinsic_matrix['pascal'].copy()
        K[:2,2]=principle
        cam_3d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        cam_3d[np.abs(cam_3d[:,2])<1e-5,2]=1e-5 # revise depth
        pts_2d=np.matmul(cam_3d,K.T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d, cam_3d

    def project_pascal_h(self, pts_3dh,RT,principle):
        K=self.intrinsic_matrix['pascal'].copy()
        K[:2,2]=principle
        return np.matmul(np.matmul(pts_3dh,RT.transpose()),K.transpose())

    @staticmethod
    def project_K(pts_3d,RT,K):
        pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
        pts_2d=np.matmul(pts_2d,K.T)
        pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
        return pts_2d

class VotingType:
    BB8=0
    BB8C=1
    BB8S=2
    VanPts=3
    Farthest=5
    Farthest4=6
    Farthest12=7
    Farthest16=8
    Farthest20=9
    SIFT=10

    @staticmethod
    def get_data_pts_2d(vote_type,data):
        if vote_type==VotingType.BB8:
            cor = data['corners'].copy()  # note the copy here!!!
            hcoords=np.concatenate([cor,np.ones([8,1],np.float32)],1) # [8,3]
        elif vote_type==VotingType.BB8C:
            cor = data['corners'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.BB8S:
            cor = data['small_bbox'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.VanPts:
            cen = data['center'].copy()
            van = data['van_pts'].copy()
            hcoords = np.concatenate([cen,np.ones([1,1],np.float32)],1)
            hcoords = np.concatenate([van,hcoords],0)
        elif vote_type==VotingType.Farthest:
            cen = data['center'].copy()
            far = data['farthest'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest4:
            cen = data['center'].copy()
            far = data['farthest4'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest12:
            cen = data['center'].copy()
            far = data['farthest12'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest16:
            cen = data['center'].copy()
            far = data['farthest16'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest20:
            cen = data['center'].copy()
            far = data['farthest20'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.SIFT:
            cen = data['center'].copy()
            far = data['sift'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)

        return hcoords

    @staticmethod
    def get_pts_3d(vote_type,class_type):
        linemod_db=LineModModelDB(class_type)
        if vote_type==VotingType.BB8C:
            points_3d = linemod_db.get_corners_3d(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.BB8S:
            points_3d = linemod_db.get_small_bbox(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.Farthest:
            points_3d = linemod_db.get_farthest_3d(class_type)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest4:
            points_3d = linemod_db.get_farthest_3d(class_type,4)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest12:
            points_3d = linemod_db.get_farthest_3d(class_type,12)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest16:
            points_3d = linemod_db.get_farthest_3d(class_type,16)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest20:
            points_3d = linemod_db.get_farthest_3d(class_type,20)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.SIFT:
            points_3d = linemod_db.get_sift_3d(class_type)
            #points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        else: # BB8
            points_3d = linemod_db.get_corners_3d(class_type)

        return points_3d

class Evaluator(object):
    def __init__(self,name):
        self.name=name
        self.linemod_db = LineModModelDB(self.name)
        self.projector=Projector()
        self.projection_2d_recorder = []
        self.add_recorder = []
        self.cm_degree_5_recorder = []
        self.proj_mean_diffs=[]
        self.add_dists=[]
        self.cm=[]
        self.degree=[]
        self.uncertainty_pnp_cost=[]

    def projection_2d(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        #print(proj_mean_diff)
        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def projection_2d_sym(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

        self.proj_mean_diffs.append(proj_mean_diff)
        self.projection_2d_recorder.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        #print(pose_pred)
        #print(pose_targets)
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        # from skimage.io import imsave
        # id=uuid.uuid1()
        # write_points('{}_pred.txt'.format(id),model_pred)
        # write_points('{}_targ.txt'.format(id),model_targets)
        #
        # img_pts_pred=pts_to_img_pts(model_pred,np.identity(3),np.zeros(3),self.projector.intrinsic_matrix['blender'])[0]
        # img_pts_pred=img_pts_to_pts_img(img_pts_pred,480,640).flatten()
        # img=np.zeros([480*640,3],np.uint8)
        # img_pts_targ=pts_to_img_pts(model_targets,np.identity(3),np.zeros(3),self.projector.intrinsic_matrix['blender'])[0]
        # img_pts_targ=img_pts_to_pts_img(img_pts_targ,480,640).flatten()
        # img[img_pts_pred>0]+=np.asarray([255,0,0],np.uint8)
        # img[img_pts_targ>0]+=np.asarray([0,255,0],np.uint8)
        # img=img.reshape([480,640,3])
        # imsave('{}.png'.format(id),img)

        mean_dist=np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def add_metric_sym(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist=np.mean(find_nearest_point_distance(model_pred,model_targets))
        self.add_recorder.append(mean_dist < diameter)
        self.add_dists.append(mean_dist)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        """ 5 cm 5 degree metric
        1. pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
        """
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cm.append(translation_distance)
        if not np.isnan(angular_distance):
            self.degree.append(angular_distance)
        self.cm_degree_5_recorder.append(translation_distance < 5 and angular_distance < 5)

    def evaluate(self, points_2d, pose_targets, class_type,index, intri_type='linemod', vote_type=VotingType.Farthest, intri_matrix=None):
        if len(index)<=4:
            return
        points_3d = VotingType.get_pts_3d(vote_type, class_type)
        p_2d=points_2d[index]
        p_3d=points_3d[index]



        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]


        pose_pred = pnp(p_3d, p_2d, K,cv2.SOLVEPNP_EPNP)


        #pose_pred = pnp(points_3d, points_2d, K,cv2.SOLVEPNP_EPNP)

        #pose_pred=ba_pnp(p_3d,p_2d,K)
        #opt_pnp(p_3d,p_2d)

        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
        #if False:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate(self, pose_pred,  pose_targets,class_type, intri_type='linemod',intri_matrix=None):

        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)
        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        if class_type in ['eggbox','glue']:
        #if False:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred


    def evaluate_uncertainty(self, mean_pts2d, covar, pose_targets, class_type,
                             intri_type='blender', vote_type=VotingType.BB8,intri_matrix=None):
        points_3d=VotingType.get_pts_3d(vote_type,class_type)

        begin=time.time()
        # full
        cov_invs = []
        for vi in range(covar.shape[0]):
            if covar[vi,0,0]<1e-6 or np.sum(np.isnan(covar)[vi])>0:
                cov_invs.append(np.zeros([2,2]).astype(np.float32))
                continue

            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = uncertainty_pnp(mean_pts2d, weights, points_3d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)
        self.uncertainty_pnp_cost.append(time.time()-begin)

        #if class_type in ['eggbox','glue']:
        if False:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_uncertainty_v2(self, mean_pts2d, covar, pose_targets, class_type,
                                intri_type='blender', vote_type=VotingType.BB8):
        points_3d = VotingType.get_pts_3d(vote_type, class_type)

        pose_pred = uncertainty_pnp_v2(mean_pts2d, covar, points_3d, self.projector.intrinsic_matrix[intri_type])
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.projection_2d_sym(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.projection_2d(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric(pose_pred, pose_targets, model, diameter)
        self.cm_degree_5_metric(pose_pred, pose_targets)

    def average_precision(self,verbose=True):
        np.save('tmp.npy',np.asarray(self.proj_mean_diffs))
        if verbose:
            print('2d projections metric: {}'.format(np.mean(self.projection_2d_recorder)))
            print('ADD metric: {}'.format(np.mean(self.add_recorder)))
            print('5 cm 5 degree metric: {}'.format(np.mean(self.cm_degree_5_recorder)))

        return np.mean(self.projection_2d_recorder),np.mean(self.add_recorder),np.mean(self.cm_degree_5_recorder)