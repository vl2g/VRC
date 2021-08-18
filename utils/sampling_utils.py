def get_iou(bb1, bb2):
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['xmin'], bb2['xmin'])
    y_top = max(bb1['ymin'], bb2['ymin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_roi_index(gt_bbox, info):
    indexes = []
    rois_info = info.item().get('bbox')
    rois = rois_info.shape[0]
    subj_roi_iou = []
    for i in range(0, rois):
        bbox_roi = rois_info[i]
        roi_bbox_dict = {}
        roi_bbox_dict["xmin"] = float(bbox_roi[0])
        roi_bbox_dict["ymin"] = float(bbox_roi[1])
        roi_bbox_dict["xmax"] = float(bbox_roi[2])
        roi_bbox_dict["ymax"] = float(bbox_roi[3])

        iou = get_iou(gt_bbox, roi_bbox_dict)
        if(iou > 0.60):
            indexes.append(i)
            subj_roi_iou.append(iou)

    return indexes, subj_roi_iou
