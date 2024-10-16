import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time
import numpy as np
from numpy.linalg import norm
import time
from yolox.utils.bbox import xyxy2cxywh,calc_IoU,Giou
from math import sqrt
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix
def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    # cost, x, y = lap.lapjv(-cost_matrix, extend_cost=True)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious_cosine(atlbrs, btlbrs,atlbrs_pred):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    # atlbrs_pred=np.array(atlbrs_pred)
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    track_vector=[(atlbrs_pred[:,0]+atlbrs_pred[:,2])/2,(atlbrs_pred[:,1]+atlbrs_pred[:,3])/2,(atlbrs[:,0]+atlbrs[:,2])/2,(atlbrs[:,1]+atlbrs[:,3])/2]
    track_vector=np.array(track_vector).T
    detection_vector=[]
    for i in range(len(atlbrs)):
        temp_v=[[((atlbrs[i,0]+atlbrs[i,2])/2)]*len(btlbrs),[((atlbrs[i,1]+atlbrs[i,3])/2)]*len(btlbrs),(btlbrs[:,0]+btlbrs[:,2])/2,(btlbrs[:,1]+btlbrs[:,3])/2]
        temp_v=np.array(temp_v).T

        features1=np.reshape(track_vector[i],(1,4))
        features2=temp_v
        norm1 = norm(features1, axis=-1).reshape(features1.shape[0], 1)
        norm2 = norm(features2, axis=-1).reshape(1, features2.shape[0])
        end_norm = np.dot(norm1, norm2)
        cos = np.dot(features1, features2.T) / end_norm
        detection_vector.append(cos[0])
    detection_vector=np.array(detection_vector)
    # detection_vector=[(atlbrs[:,0]+atlbrs[:,2])/2,(atlbrs[:,1]+atlbrs[:,3])/2,(btlbrs[:,0]+btlbrs[:,2])/2,(btlbrs[:,1]+btlbrs[:,3])/2]

    # ious = bbox_ious(
    #     np.ascontiguousarray(atlbrs, dtype=np.float),
    #     np.ascontiguousarray(btlbrs, dtype=np.float)
    # )

    return detection_vector
def ious_cosine_two(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    features1=atlbrs
    features2=btlbrs
    norm1 = norm(features1, axis=-1).reshape(features1.shape[0], 1)
    norm2 = norm(features2, axis=-1).reshape(1, features2.shape[0])
    end_norm = np.dot(norm1, norm2)
    cos = np.dot(features1, features2.T) / end_norm
    return cos[0]
def calc_shape_dev(array1):

    array1=np.array(array1)
    ious = np.zeros((len(array1)), dtype=np.float)
    if ious.size == 0:
        return ious,ious
    if len(array1)>0:
        a_c_x=(array1[:,2]+array1[:,0])/2
        a_c_y=(array1[:,3]+array1[:,1])/2
        a_c_w = (array1[:, 2] - array1[:, 0])
        a_c_h = (array1[:, 3] - array1[:, 1])
        a_s=a_c_w*a_c_h
    return a_s
def calc_shape(array1,array2):

    array1=np.array(array1)
    array2=np.array(array2)
    ious = np.zeros((len(array1), len(array2)), dtype=np.float)
    if ious.size == 0:
        return ious
    dist_array=[]
    if len(array1)>0 and len(array2)>0:
        a_c_x=(array1[:,2]+array1[:,0])/2
        a_c_y=(array1[:,3]+array1[:,1])/2
        a_c_w = (array1[:, 2] - array1[:, 0])
        a_c_h = (array1[:, 3] - array1[:, 1])

        b_c_x = (array2[:, 2] + array2[:, 0]) / 2
        b_c_y = (array2[:, 3] + array2[:, 1]) / 2
        b_c_w = (array2[:, 2] - array2[:, 0])
        b_c_h = (array2[:, 3] - array2[:, 1])
        b_s = b_c_w * b_c_h
        for i in range(len(array1)):
            a_s=a_c_w[i]*a_c_h[i]

            z_1=b_s/a_s
            all_1=z_1
            dist_array.append(all_1)
    dist_array=np.array(dist_array)
    return dist_array
def calc_dist(array1,array2):

    array1=np.array(array1)
    array2=np.array(array2)
    ious = np.zeros((len(array1), len(array2)), dtype=np.float)
    if ious.size == 0:
        return ious
    dist_array=[]
    if len(array1)>0 and len(array2)>0:
        a_c_x=(array1[:,2]+array1[:,0])/2
        a_c_y=(array1[:,3]+array1[:,1])/2
        a_c_w = (array1[:, 2] - array1[:, 0])
        a_c_h = (array1[:, 3] - array1[:, 1])
        b_c_x = (array2[:, 2] + array2[:, 0]) / 2
        b_c_y = (array2[:, 3] + array2[:, 1]) / 2
        b_c_w = (array2[:, 2] - array2[:, 0])
        b_c_h = (array2[:, 3] - array2[:, 1])
        for i in range(len(array1)):
            x_1=abs(b_c_x-a_c_x[i])
            y_1=abs(b_c_y-a_c_y[i])
            w_1 = abs(b_c_w - a_c_w[i])
            h_1 = abs(b_c_h - a_c_h[i])
            z_1=x_1*x_1+y_1*y_1
            z_1=np.sqrt(z_1)
            all_1=z_1+w_1+h_1 #########################
            temp_sqrt_a=sqrt(a_c_w[i]*a_c_w[i]+a_c_h[i]*a_c_h[i])*5
            all_1=all_1/temp_sqrt_a
            all_1=all_1/(all_1.max())
            temp_w_num=1
            temp_w_num=3
            if a_c_y[i]>7000:
                temp_w_num=temp_sqrt_a #########sota
            else:
                temp_w_num=temp_sqrt_a/4 #########sota
            temp_flagx1 = b_c_x > a_c_x[i] - temp_w_num
            temp_flagx2 = b_c_x < a_c_x[i] +  temp_w_num
            temp_flagy1 = b_c_y > a_c_y[i] -  temp_w_num
            temp_flagy2 = b_c_y < a_c_y[i] +  temp_w_num
            for t1 in range(len(all_1)):
                if temp_flagx1[t1] and temp_flagx2[t1] and temp_flagy1[t1] and temp_flagy2[t1]:
                    pass
                else:
                    all_1[t1]=1
            dist_array.append(all_1)
    dist_array=np.array(dist_array)
    return dist_array
def calc_volume_dist(bbox1,bbox2):

    bbox1=xyxy2cxywh(bbox1)
    bbox2=xyxy2cxywh(bbox2)
    h=sqrt((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2)
    v=h
    return v
def l2(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    dist=[]
    for i in range(len(atlbrs)):
        x1=atlbrs[i]
        x1_array=[]
        for j in range(len(btlbrs)):
            y1=btlbrs[j]
            l2_value=calc_volume_dist(x1,y1)
            x1_array.append(l2_value)
        dist.append(x1_array)
    dist=np.array(dist)
    return dist

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def cat_array(a1,a2,a3,a4,a5):
    ious = np.zeros((len(a1)), dtype=np.float)
    if ious.size == 0 or len(a2)==0:
        return ious
    # a1 = to_cxy(a1)
    a1 = np.array(a1)
    a2 = np.array(a2).reshape(1, len(a2))
    a3 = np.array(a3)
    a4 = np.array(a4).reshape(1, len(a4))
    a5 = np.array(a5).reshape(1, len(a5))
    #(atlbrs,atlbrs_voc_shape,atlbrs_voc,atlbrs_voc_angle,atlbrs_shape)
    a = np.hstack((a1, a2.T))
    a = np.hstack((a, a3.T))
    a = np.hstack((a, a4.T))
    a = np.hstack((a, a5.T))
    # a = a1
    return a
def cat_array2(a1,a2,a3,a4,a5):
    ious = np.zeros((len(a1)), dtype=np.float)
    if ious.size == 0 or len(a2)==0:
        return ious
    # a1 = to_cxy(a1)
    a1 = np.array(a1)
    a2 = np.array(a2).reshape(1, len(a5))
    a3 = np.array(a3).reshape(1, len(a5))
    a4 = np.array(a4).reshape(1, len(a5))
    a5 = np.array(a5).reshape(1, len(a5))

    a = np.hstack((a1, a2.T))
    a = np.hstack((a, a3.T))
    a = np.hstack((a, a4.T))
    # a = a1
    a = np.hstack((a, a5.T))
    return a
def array_to_norm(array,img_w,img_h):
    array[:, 0] /= img_w
    array[:, 2] /= img_w
    array[:, 1] /= img_h
    array[:, 3] /= img_h
    return array

def to_cxy(array):
    a=array.copy()
    a=np.array(a)
    axy = [(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2]
    axy = np.array(axy).T
    return axy
def sigmoid(x):
    y=1/(1+np.exp(-x))
    y = y + 0.5
    y = abs(1 - y)
    y = (1 - y)
    return y

def iou_distance(atracks, btracks,frame_id,history_flag=False,img_h=15556, img_w=27649):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # atlbrs_pred = [track.tlbr_pred for track in atracks]
        # atlbrs_voc = [track.get_mean_vec for track in atracks]
        # atlbrs_voc_shape = [track.get_mean_vec_shape for track in atracks]
        # atlbrs_voc_angle = [track.get_mean_vec_angle for track in atracks]
        # atlbrs_wh = [track.get_mean_wh for track in atracks]
        atlbrs = [track.tlbr for track in atracks]
        atlbrs2 = [track.tlbr2 for track in atracks]
        if history_flag:
            # btlbrs = [track.tlbr for track in btracks]
            if len(atlbrs)>0:
                # print('------------{} {} 11111111111111111111111'.format(frame_id,'iou'))
                # print(atlbrs)
                atlbrs_prdict_bbox = [track.get_predict(frame_id) for track in atracks]
                atlbrs = atlbrs_prdict_bbox

        btlbrs = [track.tlbr for track in btracks]
        btlbrs2 = [track.tlbr2 for track in btracks]
        # print('bbbbbbbbbbbbbbbbbb')
        # print(atlbrs)
        # print(btlbrs)


    # _ious = 1- ious(atlbrs, btlbrs)

    # atlbrs_wh=np.array([atlbrs_wh])
    # atlbrs_voc_h = atlbrs_voc.repeat(len(btlbrs), axis=0).T
    # atlbrs_wh = atlbrs_wh.repeat(len(btlbrs), axis=0).T
    # atlbrs_voc = np.array([atlbrs_voc])
    # atlbrs_pred = np.array(atlbrs_pred)
    # atlbrs=np.array(atlbrs)
    # btlbrs=np.array(btlbrs)
    # if len(atlbrs)>0:
    #     atlbrs = array_to_norm(atlbrs, img_w, img_h)
    # if len(btlbrs) > 0:
    #     btlbrs=array_to_norm(btlbrs,img_w,img_h)
    # if len(atlbrs_pred)>0:
    #     atlbrs_pred = array_to_norm(atlbrs_pred, img_w, img_h)
    # atlbrs_pred = np.array(atlbrs_pred)
    # atlbrs_voc = np.array(atlbrs_voc)
    # atlbrs = np.array(atlbrs)
    # btlbrs = np.array(btlbrs)
    # #
    # # if len(atlbrs) > 0 and len(btlbrs) > 0:
    # #     exit()
    #

    _ious_temp = ious(atlbrs, btlbrs)
    # #############
    _ious_shape=calc_shape(atlbrs2,btlbrs2)
    _ious_shape = 1 - _ious_shape
    _ious_shape = abs(_ious_shape)
    # temp_shape_w=sigmoid(_ious_shape)*2
    temp_shape_w=sigmoid(_ious_shape)
    _ious_temp = _ious_temp * temp_shape_w
    #########################
    # print('iiiiiiiiiiiiiiiiiiiiiiiii2222222222222222222222222222')
    # print(temp_shape_w)
    # _ious_cosine = ious_cosine(atlbrs, btlbrs,atlbrs_pred)
    _ious = 1- _ious_temp

    # _ious = 1- _ious_temp*_ious_cosine
    # _ious = _ious_temp
    cost_matrix = _ious
    return cost_matrix


    _ious_dist=calc_dist(atlbrs,btlbrs)

    _ious_shape=calc_shape(atlbrs,btlbrs)
    atlbrs_shape=calc_shape_dev(atlbrs)
    btlbrs_shape=calc_shape_dev(btlbrs)
    # a
    a_all=cat_array(atlbrs,atlbrs_voc_shape,atlbrs_voc,atlbrs_voc_angle,atlbrs_shape)
    # a_all=cat_array(atlbrs_cxy,atlbrs_voc_shape,atlbrs_voc,atlbrs_voc_angle,atlbrs_shape)
    ious_1 = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious_1.size == 0:
        out_cxc=ious_1
    else:
        out_cxc = []
        # print('-------------------------------------------==========================================')
        for i in range(len(a_all)):
            # a1 a2 a3 a4 a5
            b_all = cat_array2(btlbrs, _ious_shape[i], _ious_dist[i], _ious_cosine[i], btlbrs_shape)
            # b_all = cat_array2(btlbrs_cxy, _ious_shape[i], _ious_dist[i], _ious_cosine[i], btlbrs_shape)
            temp_a_all=np.array([a_all[i]])
            # print('-------------------------------------------')
            # print('aaaaaaaaaaaaaaaa')
            # print(temp_a_all)
            # print('bbbbbbbbbbb')
            # print(b_all)
            all_cosine=ious_cosine_two(temp_a_all,b_all)
            # print('cccccccccccccc')
            # print(all_cosine)
            out_cxc.append(all_cosine)

        out_cxc = np.array(out_cxc)


    # cost_matrix=1-out_cxc
    # print('oooooooooooooooooooo')
    # print(out_cxc)

    # cost_matrix=1-(out_cxc+_ious_temp)/2
    # dist_voc = np.where(dist_voc_current>200, 1000000000, dist_voc)
    # temp_out_cost=np.where(_ious_temp<=0.1,out_cxc,_ious_temp)
    # temp_out_cost=np.where(_ious_temp==0,out_cxc,_ious_temp)
    temp_out_cost=np.where(_ious_temp==0,_ious_dist,_ious_temp)
    # temp_out_cost=np.where(_ious_temp==0,_ious_cosine,_ious_temp)
    # temp_out_cost=_ious_cosine*_ious_dist
    cost_matrix=1-temp_out_cost
    # cost_matrix=_ious_temp+1-np.abs(_ious_dist*(2-_ious_cosine))
    cost_matrix=_ious
    # cost_matrix=_ious
    # cost_matrix=temp_out_cost
    return cost_matrix
def dist_distance(atracks, btracks,frame_id=1,history_flag=False,img_h=15556, img_w=27649):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    norm_size=np.sqrt(img_w*img_h)*2
    # norm_size=np.sqrt(img_w*img_h)
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # atlbrs_pred = [track.tlbr_pred for track in atracks]
        # atlbrs_voc = [track.get_mean_vec for track in atracks]
        # atlbrs_voc_shape = [track.get_mean_vec_shape for track in atracks]
        # atlbrs_voc_angle = [track.get_mean_vec_angle for track in atracks]
        # atlbrs_wh = [track.get_mean_wh for track in atracks]
        atlbrs = [track.tlbr2 for track in atracks]
        if history_flag:
            if len(atlbrs) > 0:
                atlbrs_prdict_bbox = [track.get_predict(frame_id) for track in atracks]
                atlbrs = atlbrs_prdict_bbox
        # atlbrs_prdict_bbox = [track.get_predict for track in atracks]
        btlbrs = [track.tlbr2 for track in btracks]
        # atlbrs=atlbrs_prdict_bbox


    # # _ious = 1- ious(atlbrs, btlbrs)
    #
    # # atlbrs_wh=np.array([atlbrs_wh])
    # # atlbrs_voc_h = atlbrs_voc.repeat(len(btlbrs), axis=0).T
    # # atlbrs_wh = atlbrs_wh.repeat(len(btlbrs), axis=0).T
    # atlbrs_pred = np.array(atlbrs_pred)
    # atlbrs_voc = np.array([atlbrs_voc])
    # atlbrs=np.array(atlbrs)
    # btlbrs=np.array(btlbrs)
    # # if len(atlbrs)>0:
    # #     atlbrs = array_to_norm(atlbrs, img_w, img_h)
    # # if len(btlbrs) > 0:
    # #     btlbrs=array_to_norm(btlbrs,img_w,img_h)
    # # if len(atlbrs_pred)>0:
    # #     atlbrs_pred = array_to_norm(atlbrs_pred, img_w, img_h)
    # atlbrs_pred = np.array(atlbrs_pred)
    # atlbrs_voc = np.array(atlbrs_voc)
    # atlbrs = np.array(atlbrs)
    # btlbrs = np.array(btlbrs)
    # #
    # # if len(atlbrs) > 0 and len(btlbrs) > 0:
    # #     exit()



    _ious_dist=calc_dist(atlbrs,btlbrs)
    # _ious_dist/=norm_size
    # _ious_shape=calc_shape(atlbrs,btlbrs)
    # _ious_shape = 1 - _ious_shape
    # _ious_shape = abs(_ious_shape)
    # temp_shape_w=sigmoid(_ious_shape)
    # # temp_shape_w=1-temp_shape_w
    # _ious_dist = _ious_dist * temp_shape_w



    return _ious_dist

    _ious_shape=calc_shape(atlbrs,btlbrs)
    atlbrs_shape=calc_shape_dev(atlbrs)
    btlbrs_shape=calc_shape_dev(btlbrs)
    # a
    a_all=cat_array(atlbrs,atlbrs_voc_shape,atlbrs_voc,atlbrs_voc_angle,atlbrs_shape)
    # a_all=cat_array(atlbrs_cxy,atlbrs_voc_shape,atlbrs_voc,atlbrs_voc_angle,atlbrs_shape)
    ious_1 = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious_1.size == 0:
        out_cxc=ious_1
    else:
        out_cxc = []
        # print('-------------------------------------------==========================================')
        for i in range(len(a_all)):
            # a1 a2 a3 a4 a5
            b_all = cat_array2(btlbrs, _ious_shape[i], _ious_dist[i], _ious_cosine[i], btlbrs_shape)
            # b_all = cat_array2(btlbrs_cxy, _ious_shape[i], _ious_dist[i], _ious_cosine[i], btlbrs_shape)
            temp_a_all=np.array([a_all[i]])
            # print('-------------------------------------------')
            # print('aaaaaaaaaaaaaaaa')
            # print(temp_a_all)
            # print('bbbbbbbbbbb')
            # print(b_all)
            all_cosine=ious_cosine_two(temp_a_all,b_all)
            # print('cccccccccccccc')
            # print(all_cosine)
            out_cxc.append(all_cosine)

        out_cxc = np.array(out_cxc)


    # cost_matrix=1-out_cxc
    # print('oooooooooooooooooooo')
    # print(out_cxc)

    # cost_matrix=1-(out_cxc+_ious_temp)/2
    # dist_voc = np.where(dist_voc_current>200, 1000000000, dist_voc)
    # temp_out_cost=np.where(_ious_temp<=0.1,out_cxc,_ious_temp)
    # temp_out_cost=np.where(_ious_temp==0,out_cxc,_ious_temp)
    temp_out_cost=np.where(_ious_temp==0,_ious_dist,_ious_temp)
    # temp_out_cost=np.where(_ious_temp==0,_ious_cosine,_ious_temp)
    # temp_out_cost=_ious_cosine*_ious_dist
    cost_matrix=1-temp_out_cost
    # cost_matrix=_ious_temp+1-np.abs(_ious_dist*(2-_ious_cosine))
    cost_matrix=_ious
    # cost_matrix=_ious
    # cost_matrix=temp_out_cost
    return cost_matrix
def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix



def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def get_image_feat(tracks, detections,metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    track_features = np.asarray([track.image_feat for track in tracks], dtype=np.float)
    det_features = np.asarray([track.image_feat for track in detections], dtype=np.float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    # cost_matrix=[]
    # for i in range(len(track_features)):
    #     temp_c=abs(det_features-track_features[i])
    #     temp_c = np.clip(temp_c, 0, 100)
    #     cost_matrix.append(temp_c)

    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_feat(cost_matrix, dists):
    cost_matrix=np.array(cost_matrix)
    ious = np.zeros(cost_matrix.shape, dtype=np.float)
    if ious.size == 0:
        return ious
    iou_dists = 1 - dists
    iou_dists=iou_dists*cost_matrix
    fuse_cost=1-iou_dists

    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost