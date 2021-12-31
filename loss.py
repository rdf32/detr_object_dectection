from scipy.optimize import linear_sum_assignment
from typing import Union,Dict,Tuple
import matplotlib.pyplot as plt
import random
import cv2

def get_inp_shape(dataset):
    
    for i, (image, t_bbox, t_labels) in enumerate(dataset):
        input_shape = image.shape
        if i > 1: break
    return np.array(image.shape)

inp_shape = get_inp_shape(train_dataset)

def intersect(box_a: tf.Tensor, box_b: tf.Tensor) -> tf.Tensor:
    """
    Compute the intersection area between two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The intersection area [a, b] between each bbox. zero if no intersection
    """
    # resize both tensors to [A,B,2] with the tile function to compare
    # each bbox with the anchors:
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    # Then we compute the area of intersect between box_a and box_b.
    # box_a: (tensor) bounding boxes, Shape: [n, A, 4].
    # box_b: (tensor) bounding boxes, Shape: [n, B, 4].
    # Return: (tensor) intersection area, Shape: [n,A,B].

    A = tf.shape(box_a)[0] # Number of possible bbox
    B = tf.shape(box_b)[0] # Number of anchors

    #print(A, B, box_a.shape, box_b.shape)
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymax = tf.tile(tf.expand_dims(box_a[:, 2:], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymax = tf.tile(tf.expand_dims(box_b[:, 2:], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    above_right_corner = tf.math.minimum(tiled_box_a_xymax, tiled_box_b_xymax)


    # Upper Left Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymin = tf.tile(tf.expand_dims(box_a[:, :2], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymin = tf.tile(tf.expand_dims(box_b[:, :2], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    upper_left_corner = tf.math.maximum(tiled_box_a_xymin, tiled_box_b_xymin)


    # If there is some intersection, both must be > 0
    inter = tf.nn.relu(above_right_corner - upper_left_corner)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor, return_union=False) -> tf.Tensor:
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The Jaccard overlap [a, b] between each bbox
    """
    # Get the intersectin area
    inter = intersect(box_a, box_b)

    # Compute the A area
    # (xmax - xmin) * (ymax - ymin)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    # Tile the area to match the anchors area
    area_a = tf.tile(tf.expand_dims(area_a, axis=-1), [1, tf.shape(inter)[-1]])

    # Compute the B area
    # (xmax - xmin) * (ymax - ymin)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    # Tile the area to match the gt areas
    area_b = tf.tile(tf.expand_dims(area_b, axis=-2), [tf.shape(inter)[-2], 1])

    union = area_a + area_b - inter

    if return_union is False:
        # Return the intesect over union
        return inter / union
    else:
        return inter / union, union

def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Merged two set of boxes so that operations ca be run to compare them
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        Return the two same tensor tiled: (a, b, 4)
    """
    A = tf.shape(box_a)[0] # Number of bbox in box_a
    B = tf.shape(box_b)[0] # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])

    return tiled_box_a, tiled_box_b

def np_tf_linear_sum_assignment(matrix):

    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    #print(matrix.shape, target_indices, pred_indices)

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    return [target_indices, pred_indices, target_selector, pred_selector]

def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy

def hungarian_matching(t_bbox, t_class, p_bbox, p_class, fcost_class=1, fcost_bbox=5, fcost_giou=2, slice_preds=True) -> tuple:

    if slice_preds:
        size = tf.cast(t_bbox[0][0], tf.int32)
        t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
        t_class = tf.slice(t_class, [1, 0], [size, -1])
        t_class = tf.squeeze(t_class, axis=-1)

    # Convert frpm [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)

    softmax = tf.nn.softmax(p_class)

    # Classification cost for the Hungarian algorithom 
    # On each prediction. We select the prob of the expected class
    cost_class = -tf.gather(softmax, t_class, axis=1)

    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    cost_giou = -(iou - (area - union) / area)

    # Final hungarian cost matrix
    cost_matrix = fcost_bbox * cost_bbox + fcost_class * cost_class + fcost_giou * cost_giou

    selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool] )
    target_indices = selectors[0]
    pred_indices = selectors[1]
    target_selector = selectors[2]
    pred_selector = selectors[3]

    return pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class

@tf.function
def get_total_losss(losses):
    """
    Get model total losss including auxiliary loss
    """
    train_loss = ["label_cost", "giou_loss", "l1_loss"]
    loss_weights = [1, 2, 5]
    iou = losses['iou']

    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_loss) if loss_name in key]
        if len(selector) == 1:

            total_loss += losses[key]*loss_weights[selector[0]]
    return total_loss, iou


def get_losses(m_outputs, t_bbox, t_class):
    losses = get_detr_losses(m_outputs, t_bbox, t_class)
    # Compute the total loss
    total_loss = get_total_losss(losses)

    return total_loss #, losses


def loss_labels(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector, background_class=0):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.zeros((tf.shape(neg_p_class)[0],), tf.int64) + background_class
    
    neg_weights = tf.zeros((tf.shape(neg_indices)[0],)) + 0.1
    pos_weights = tf.zeros((tf.shape(t_indices)[0],)) + 1.0
    weights = tf.concat([neg_weights, pos_weights], axis=0)
    
    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)

    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)

    return loss
def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def loss_boxes(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector):

    p_bbox = tf.gather(p_bbox, p_indices)
    t_bbox = tf.gather(t_bbox, t_indices)


    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)

    l1_loss = tf.abs(p_bbox-t_bbox)
    l1_loss = tf.reduce_sum(l1_loss) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)

    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    giou = (iou - (area - union) / area)
    loss_giou = 1 - tf.linalg.diag_part(giou)
    
    iou = compute_iou(p_bbox, t_bbox)
    loss_giou = tf.reduce_sum(loss_giou) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    return loss_giou, l1_loss, iou

def get_detr_losses(m_outputs, target_bbox, target_label, suffix=""):

    predicted_bbox = m_outputs["pred_boxes"]
    predicted_label = m_outputs["pred_logits"]

    all_target_bbox = []
    all_target_class = []
    all_predicted_bbox = []
    all_predicted_class = []
    all_target_indices = []
    all_predcted_indices = []
    all_target_selector = []
    all_predcted_selector = []

    t_offset = 0
    p_offset = 0

    for b in range(inp_shape[0]):

        p_bbox, p_class, t_bbox, t_class = predicted_bbox[b], predicted_label[b], target_bbox[b], target_label[b]
        t_indices, p_indices, t_selector, p_selector, t_bbox, t_class = hungarian_matching(t_bbox, t_class, p_bbox, p_class, slice_preds=True)

        t_indices = t_indices + tf.cast(t_offset, tf.int64)
        p_indices = p_indices + tf.cast(p_offset, tf.int64)

        all_target_bbox.append(t_bbox)
        all_target_class.append(t_class)
        all_predicted_bbox.append(p_bbox)
        all_predicted_class.append(p_class)
        all_target_indices.append(t_indices)
        all_predcted_indices.append(p_indices)
        all_target_selector.append(t_selector)
        all_predcted_selector.append(p_selector)

        t_offset += tf.shape(t_bbox)[0]
        p_offset += tf.shape(p_bbox)[0]

    all_target_bbox = tf.concat(all_target_bbox, axis=0)
    all_target_class = tf.concat(all_target_class, axis=0)
    all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
    all_predicted_class = tf.concat(all_predicted_class, axis=0)
    all_target_indices = tf.concat(all_target_indices, axis=0)
    all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
    all_target_selector = tf.concat(all_target_selector, axis=0)
    all_predcted_selector = tf.concat(all_predcted_selector, axis=0)


    label_cost = loss_labels(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
        background_class=0,
    )

    giou_loss, l1_loss, iou = loss_boxes(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss
    iou = iou

    return {
        "label_cost{}".format(suffix): label_cost,
        "giou_loss{}".format(suffix): giou_loss,
        "l1_loss{}".format(suffix): l1_loss,
        "iou{}".format(suffix): iou
    }