import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from yolox.utils.bbox import xyxy2cxywh,calc_IoU,Giou,xywh2cxywh,xywh2xyxy
from .kalman_filter import KalmanFilter
import time
from yolox.tracker import matching_gigatracker as matching
from .basetrack import BaseTrack, TrackState
from math import sqrt
import warnings

from fast_reid.fast_reid_interfece import FastReIDInterface
warnings.filterwarnings('ignore')

def array_to_norm(array,img_w,img_h):
    array[0] /= img_w
    array[2] /= img_w
    array[1] /= img_h
    array[3] /= img_h
    return array
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score,score2,score4,score5,idx, cls=None,feat=None, buffer_size=10):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.mean_orgin=None
        self.is_activated = False
        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.score2 = score2
        self.score4 = score4
        self.score5 = score5
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.idx = idx
        # self.is_activated = True
        # self.image_feat=temp_feat

        self.tracklet_len = 0
        self.previous_bbox=self._tlwh+0.01
        self.history_bbox=[]
        self.history_bbox_id=[]
        self.history_bbox_w=[]
        self.history_bbox_h=[]
        self.history_bbox_cx=[]
        self.history_bbox_cy=[]
        self.history_bbox_cx=np.append(self.history_bbox_cx,tlwh[0]+tlwh[2]/2)
        self.history_bbox_cy=np.append(self.history_bbox_cy,tlwh[1]+tlwh[3]/2)
        self.history_bbox_w=np.append(self.history_bbox_w,tlwh[2])
        self.history_bbox_h=np.append(self.history_bbox_h,tlwh[3])
        self.predict_bbox=[tlwh[0],tlwh[1],tlwh[0]+tlwh[2],tlwh[1]+tlwh[3]]
        self.history_bbox_id=np.append(self.history_bbox_id,0)

        # self.history_bbox.append(self._tlwh)
        # self.history_bbox=np.append(self.history_bbox,self._tlwh)
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance, = self.kalman_filter.predict(mean_state, self.covariance)
        # print('pppppppppppppppppppppppppppppppppp')
        # print(self.mean)


    def predict_w(self,frame_id):
        if len(self.history_bbox_w)>=2:
            z1=np.polyfit(self.history_bbox_id,self.history_bbox_w,2)
            p1=np.poly1d(z1)
            return p1(frame_id)
        else:
            return self.history_bbox_w[-1]

    def predict_h(self,frame_id):
        if len(self.history_bbox_h)>=2:
            z1=np.polyfit(self.history_bbox_id,self.history_bbox_h,2)
            p1=np.poly1d(z1)
            return p1(frame_id)
        else:
            return self.history_bbox_h[-1]

    def predict_cx(self, frame_id):
        if len(self.history_bbox_cx) >= 2:
            z1 = np.polyfit(self.history_bbox_id, self.history_bbox_cx,2)
            p1 = np.poly1d(z1)
            return p1(frame_id)
        else:
            return self.history_bbox_cx[-1]

    def predict_cy(self, frame_id):
        if len(self.history_bbox_cy) >= 2:
            z1 = np.polyfit(self.history_bbox_id, self.history_bbox_cy,2)
            p1 = np.poly1d(z1)
            return p1(frame_id)
        else:
            return self.history_bbox_cy[-1]

    def get_predict_bbox(self,frame_id):
        cx=self.predict_cx(frame_id)
        cy=self.predict_cy(frame_id)
        w=self.predict_w(frame_id)
        h=self.predict_h(frame_id)
        x1=cx-w/2
        y1=cy-h/2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1,y1,x2,y2]



    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean_orgin=self.mean.copy()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        temp_iou_thre=0.75
        if  True :
            if self.score>temp_iou_thre:
                self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # self.previous_bbox = self.tlwh_to_xyah(self._tlwh)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if self.kalman_flag:
            self.mean[:4] = self.tlwh_to_xyah(new_track.tlwh)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.score2 = new_track.score2
        self.score4 = new_track.score
        self.idx = new_track.idx

        if len(self.history_bbox)==0:
            self.history_bbox=[self.tlwh]
        else:
            self.history_bbox=np.array(self.history_bbox)
            temp_add=np.array([self.tlwh])
            self.history_bbox=np.vstack((self.history_bbox,temp_add))
            # self.history_bbox=np.append(self.history_bbox,[self._tlwh])q
        self.history_bbox_id=np.append(self.history_bbox_id,frame_id)
        self.history_bbox_w=np.append(self.history_bbox_w,self.tlwh[2])
        self.history_bbox_h=np.append(self.history_bbox_h,self.tlwh[3])
        self.history_bbox_cx=np.append(self.history_bbox_cx,self.tlwh[0]+self.tlwh[2]/2)
        self.history_bbox_cy=np.append(self.history_bbox_cy,self.tlwh[1]+self.tlwh[3]/2)
        if len(self.history_bbox)>5:
            self.history_bbox=np.delete(self.history_bbox,0,axis=0)
            self.history_bbox_id=np.delete(self.history_bbox_id,0)
            self.history_bbox_h=np.delete(self.history_bbox_h,0)
            self.history_bbox_w=np.delete(self.history_bbox_w,0)
            self.history_bbox_cx=np.delete(self.history_bbox_cx,0)
            self.history_bbox_cy=np.delete(self.history_bbox_cy,0)



        # self.previous_bbox = self.tlwh_to_xyah(new_track.tlwh)
        # self.previous_id = frame_id

    def update_previous(self, new_track, frame_id):
        # self.previous_bbox=self.tlwh_to_xyah(new_track.tlwh)
        self.previous_bbox=new_track.tlwh

        self.previous_id=frame_id


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        if self.kalman_flag:
            self.mean[:4] = self.tlwh_to_xyah(new_track.tlwh)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.score2 = new_track.score2
        self.score4 = new_track.score4
        self.idx = new_track.idx


        if len(self.history_bbox)==0:
            self.history_bbox=[self.tlwh]
        else:
            self.history_bbox=np.array(self.history_bbox)
            temp_add=np.array([self.tlwh])
            self.history_bbox=np.vstack((self.history_bbox,temp_add))
            # self.history_bbox=np.append(self.history_bbox,[self._tlwh])q
        self.history_bbox_id=np.append(self.history_bbox_id,frame_id)
        self.history_bbox_w=np.append(self.history_bbox_w,self.tlwh[2])
        self.history_bbox_h=np.append(self.history_bbox_h,self.tlwh[3])
        self.history_bbox_cx=np.append(self.history_bbox_cx,self.tlwh[0]+self.tlwh[2]/2)
        self.history_bbox_cy=np.append(self.history_bbox_cy,self.tlwh[1]+self.tlwh[3]/2)
        if len(self.history_bbox)>5:
            self.history_bbox=np.delete(self.history_bbox,0,axis=0)
            self.history_bbox_id=np.delete(self.history_bbox_id,0)
            self.history_bbox_h=np.delete(self.history_bbox_h,0)
            self.history_bbox_w=np.delete(self.history_bbox_w,0)
            self.history_bbox_cx=np.delete(self.history_bbox_cx,0)
            self.history_bbox_cy=np.delete(self.history_bbox_cy,0)


    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        temp_n=5
        count_ge=0
        temp_num = (self.all_h[int(len(self.all_h) / 1.5)])
        ret=xywh2cxywh(ret)
        ret=[ret[0]-temp_num,ret[1]-temp_num,ret[0]+temp_num,ret[1]+temp_num]
        return ret

    @property
    # @jit(nopython=True)
    def tlbr2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        # ret[2:] += ret[:2]
        return ret
    @property
    # @jit(nopython=True)
    def get_mean_vec(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.history_bbox.copy()
        ret_id = self.history_bbox_id.copy()
        voc=[]
        if len(ret)<=1:
            return 0.00001
        else:
            for i in range(len(ret)-1):
                pre_bbox=ret[i]
                pre_bbox_id=ret_id[i]
                pre_bbox=xywh2cxywh(pre_bbox)
                after_bbox=ret[i+1]
                after_bbox_id=ret_id[i+1]
                # iou_value=calc_IoU(xywh2xyxy(pre_bbox),xywh2xyxy(after_bbox))
                after_bbox=xywh2cxywh(after_bbox)
                after_bbox=array_to_norm(after_bbox,self.img_w,self.img_h)
                pre_bbox=array_to_norm(pre_bbox,self.img_w,self.img_h)
                l2_val=sqrt((after_bbox[0]-pre_bbox[0])**2+(after_bbox[1]-pre_bbox[1])**2)
                # voc.append((l2_val/(after_bbox_id-pre_bbox_id)))
                voc.append((l2_val/(after_bbox_id-pre_bbox_id)))
            return np.mean(voc)

    @property
    # @jit(nopython=True)
    def get_mean_vec_shape(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.history_bbox.copy()
        ret_id = self.history_bbox_id.copy()
        voc = []
        if len(ret) <= 1:
            return 1
        else:
            for i in range(len(ret) - 1):
                pre_bbox = ret[i]
                pre_bbox_id = ret_id[i]
                after_bbox = ret[i + 1]
                after_bbox_id = ret_id[i + 1]
                # iou_value = calc_IoU(xywh2xyxy(pre_bbox), xywh2xyxy(after_bbox))
                # pre_bbox = xywh2cxywh(pre_bbox)
                # after_bbox = xywh2cxywh(after_bbox)
                l2_val = (after_bbox[2] * after_bbox[3]) / (pre_bbox[2] * pre_bbox[3])
                # voc.append((l2_val/(after_bbox_id-pre_bbox_id)))

                voc.append((l2_val))
            return np.mean(voc)

    @property
    # @jit(nopython=True)
    def get_mean_vec_angle(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.history_bbox.copy()
        ret_id = self.history_bbox_id.copy()
        voc = []
        if len(ret) <= 2:
            return 1
        else:
            p1,p2,p3=ret[-1],ret[-2],ret[-3]
            p1=xyxy2cxywh(p1)
            p2=xyxy2cxywh(p2)
            p3=xyxy2cxywh(p3)
            vec1=[p3[0],p3[1],p2[0],p2[1]]
            vec2=[p2[0],p2[1],p1[0],p1[1]]
            dist1 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return dist1

    @property
    # @jit(nopython=True)
    def get_mean_wh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret_w = self.history_bbox_w.copy()
        ret_h = self.history_bbox_h.copy()
        return sqrt(np.mean(ret_w)**2+np.mean(ret_h)**3)

    # @property
    # @jit(nopython=True)
    def get_predict(self,frame_id):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if len(self.history_bbox_id)>1:
            return self.get_predict_bbox(frame_id)
        else:
            return [self.history_bbox_cx[-1]-self.history_bbox_w[-1]/2,self.history_bbox_cy[-1]-self.history_bbox_h[-1]/2,self.history_bbox_cx[-1]+self.history_bbox_w[-1]/2,self.history_bbox_cy[-1]+self.history_bbox_h[-1]/2]

    @property
    # @jit(nopython=True)
    def get_last_id(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.history_bbox_id[-1]

    @property
    # @jit(nopython=True)
    def tlbr_pred(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.previous_bbox.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    def set_all_h(self,all_h):
        self.all_h=all_h
        self.all_h.sort()

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class GigaTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        STrack.kalman_flag = args.kalman_flag
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        if args.with_reid:

            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        try:
            self.dist_flag = args.dist_flag
        except:
            self.dist_flag = False
        try:
            self.args.l2_thre_num = args.l2_thre_num
        except:
            self.args.l2_thre_num = 100000

        self.det_thresh = args.track_thresh + 0.1
        # self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results,img_info,dist_flag=False,history_flag=False,l2_thre_num=10000,obj=None,args=None,img=None,save_reid_feature_flag=False):
        self.frame_id += 1
        # BaseTrack._count=0
        self.obj=obj
        STrack.obj=obj
        self.dist_flag=dist_flag
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []


        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            scores2 = output_results[:, 4]
            scores4 = output_results[:, 4]
            scores5 = output_results[:, 5]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            scores4 = output_results[:, 4]
            scores5 = output_results[:, 5]
            scores2 = np.minimum(output_results[:, 4] , output_results[:, 5])
            classes = output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # bboxes /= scale
        all_index=[t for t in range(len(output_results))]
        if self.args.with_reid:
            dets=bboxes
            features_all = self.encoder.inference(img, dets)

        STrack.img_w = img_info['width']
        STrack.img_h = img_info['height']
        img_w=img_info['width']
        img_h=img_info['height']
        all_index=np.array(all_index)
        all_h=output_results[:,3]-output_results[:,1]
        all_w=output_results[:,2]-output_results[:,0]
        all_h.sort()
        all_w.sort()
        STrack.all_h = all_h
        STrack.all_w = all_w
        STrack.img_h = img_h
        STrack.img_w = img_w

        remain_inds = scores >= self.args.track_thresh
        inds_low = scores2 > 0.05
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        all_index=[i1 for i1 in range(len(output_results))]
        all_index=np.array(all_index)
        index_array=[]
        index_array_second=[]
        dets_second = bboxes[inds_second]
        # index_array_second = all_index[inds_second]
        index_array_second = all_index[inds_second]
        dets = bboxes[remain_inds]
        classes_keep = classes[remain_inds]
        index_array = all_index[remain_inds]
        scores_keep = scores[remain_inds]
        scores_keep2 = scores2[remain_inds]
        scores_keep4 = scores4[remain_inds]
        scores_keep5 = scores5[remain_inds]
        if args.with_reid:
            features_keep = features_all[remain_inds]
            features_second = features_all[inds_second]
        scores_second = scores[inds_second]
        scores_second2 = scores2[inds_second]
        scores_second4 = scores4[inds_second]
        scores_second5 = scores5[inds_second]
        classes_second = classes[inds_second]

        iou_thre = 0.8
        match_thre = 0.8
        tic1=time.time()
        if len(dets) > 0:
            '''Detections'''
            if args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s,s2, s4,s5,idx,c,f) for
                              (tlbr, s,s2, s4,s5,idx,c,f) in zip(dets, scores_keep,scores_keep2,scores_keep4,scores_keep5,index_array,classes_keep,features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, s2, s4,s5, idx, c) for
                              (tlbr, s, s2, s4, s5, idx, c) in zip(dets, scores_keep,scores_keep2,scores_keep4, scores_keep5,index_array, classes_keep)]
        else:
            detections = []
        toc1=time.time()
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated :
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        tic1 = time.time()

        if self.dist_flag:
            dists = matching.dist_distance(strack_pool, detections,self.frame_id,history_flag,img_h, img_w)
            matches, u_track, u_detection, = matching.linear_assignment(dists, thresh=self.args.l2_thre_num)
        else:
            tic2 = time.time()
            ious_dists = matching.iou_distance(strack_pool, detections,self.frame_id, history_flag,img_h, img_w)
            toc2=time.time()
            tic2=time.time()

            dists_dist = matching.dist_distance(strack_pool, detections,self.frame_id,history_flag,img_h, img_w)
            toc2 = time.time()
            dist_dists_mask = (dists_dist > iou_thre)
            tic2 = time.time()

            if not self.args.mot20:
                # pass
                ious_dists = matching.fuse_score(ious_dists, detections)
                dists_dist = matching.fuse_score(dists_dist, detections)
            toc2 = time.time()
            tic2 = time.time()

            dists_dist[ious_dists < iou_thre] = 1
            # ious_dists[dist_dists_mask] = 1
            ious_dists[ious_dists >=iou_thre] = 1
            dists_dist += iou_thre
            #
            # dists=iou_thre*ious_dists+(1-iou_thre)*dists_dist
            dists = np.minimum(ious_dists, dists_dist)
            # dists = ious_dists
            toc2=time.time()
            if self.args.with_reid:
                emb_dists = matching.embedding_distance(strack_pool, detections)
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > np.mean(emb_dists)] = 1.0
                # emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[dist_dists_mask] = 1.0
                iou_thre2 = 0.7 ####
                dists = dists * iou_thre2 + (1 - iou_thre2) * emb_dists

            matches, u_track, u_detection, = matching.linear_assignment(dists, thresh=match_thre) ##########thr#############

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update_previous(strack_pool[itracked], self.frame_id-1)
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.update_previous(track, self.frame_id - 1)
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if args.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s,s2,s4,s5,idx,c,f) for
                              (tlbr, s,s2,s4,s5,idx,c,f) in zip(dets_second, scores_second, scores_second2, scores_second4,scores_second5,index_array_second,classes_second,features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s,s2,s4,s5, idx, c) for
                                     (tlbr, s,s2, s4,s5, idx, c) in
                                     zip(dets_second, scores_second, scores_second2,scores_second4,scores_second5, index_array_second, classes_second
                                        )]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if self.dist_flag:
            dists = matching.dist_distance(r_tracked_stracks, detections_second,self.frame_id+1,history_flag, img_h, img_w)
            matches, u_track, u_detection_second, = matching.linear_assignment(dists, thresh=l2_thre_num)
        else:
            ious_dists = matching.iou_distance(r_tracked_stracks, detections_second,self.frame_id+1,history_flag,img_h, img_w)

            dists_dist = matching.dist_distance(r_tracked_stracks, detections_second, self.frame_id, history_flag, img_h, img_w)
            dist_dists_mask = (dists_dist > iou_thre)
            if not self.args.mot20:
                # pass
                ious_dists = matching.fuse_score(ious_dists, detections_second)
                dists_dist = matching.fuse_score(dists_dist, detections_second)

            dists_dist[ious_dists < iou_thre] = 1
            ious_dists[ious_dists >= iou_thre] = 1
            # ious_dists[dist_dists_mask] = 1
            dists_dist += iou_thre
            #
            # dists=iou_thre*ious_dists+(1-iou_thre)*dists_dist
            dists = np.minimum(ious_dists, dists_dist)
            # dists=ious_dists

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(r_tracked_stracks, detections_second)
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > np.mean(emb_dists)] = 1.0
                # emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[dist_dists_mask] = 1.0
                iou_thre2 = 0.6
                ious_dists=dists*iou_thre2+(1-iou_thre2)*emb_dists
            dists = ious_dists
            matches, u_track, u_detection_second, = matching.linear_assignment(dists, thresh=0.65)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_previous(track, self.frame_id - 1)
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.update_previous(track, self.frame_id - 1)
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        toc1 = time.time()

        tic1 = time.time()

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

        temp_iou_thre=0.6

        detections = [detections[i] for i in u_detection if detections[i].score>temp_iou_thre]
        detections2 = [detections_second[i] for i in u_detection_second if detections_second[i].score4>temp_iou_thre   ] ####################
        detections=detections+detections2
        if self.dist_flag:
            dists = matching.dist_distance(unconfirmed, detections,self.frame_id+1,history_flag, img_h, img_w)
            matches, u_unconfirmed, u_detection, = matching.linear_assignment(dists, thresh=self.args.l2_thre_num)
        else:
            ious_dists = matching.iou_distance(unconfirmed, detections,self.frame_id+1,history_flag,img_h, img_w)
            dists_dist = matching.dist_distance(unconfirmed, detections, self.frame_id + 1, history_flag, img_h, img_w)
            dist_dists_mask = (dists_dist > iou_thre)
            if not self.args.mot20:
                ious_dists = matching.fuse_score(ious_dists, detections)
                dists_dist = matching.fuse_score(dists_dist, detections)


            dists_dist[ious_dists < iou_thre] = 1
            ious_dists[ious_dists >= iou_thre] = 1
            # ious_dists[dist_dists_mask] = 1
            dists_dist += iou_thre
            dists=np.minimum(ious_dists,dists_dist)
            if self.args.with_reid:
                emb_dists = matching.embedding_distance(unconfirmed, detections)
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > np.mean(emb_dists)] = 1.0
                # emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[dist_dists_mask] = 1.0
                iou_thre2=0.7
                dists=dists*iou_thre2+(1-iou_thre2)*emb_dists
            matches, u_unconfirmed, u_detection, = matching.linear_assignment(dists, thresh=match_thre) ###########thr############
        for itracked, idet in matches:
            unconfirmed[itracked].update_previous(unconfirmed[itracked], self.frame_id-1)
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        toc1 = time.time()
        tic1 = time.time()
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if  track.score < self.det_thresh :
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks,self.lost_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb,frame_id,img_h, img_w,dist_flag=False):
    if dist_flag:
        pdist = matching.dist_distance(stracksa, stracksb,frame_id,img_h, img_w)
    else:
        pdist = matching.iou_distance(stracksa, stracksb,frame_id,img_h, img_w)
    # pairs = np.where(pdist < 0.15)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
