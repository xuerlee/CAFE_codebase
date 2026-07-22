import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path

import json
import os
import re
from collections import defaultdict


# Action_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
#                 'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
#                 'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
#                 'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
#                 'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
#                 'writing', 'lying', 'pulling']
# Activity_names = ['standing', 'walking', 'sitting', 'holding sth', 'listening to someone',
#                   'talking to someone', 'looking at robot', 'looking into sth', 'cycling',
#                   'looking at sth', 'going upstairs', 'bending', 'typing', 'interaction with door',
#                   'eating sth', 'talking on the phone', 'going downstairs', 'scootering',
#                   'pointing at sth', 'pushing', 'reading', 'skating', 'running', 'greeting gestures',
#                   'writing', 'lying', 'pulling']
Action_names = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running', 'lying']
Activity_names = ['walking', 'standing', 'sitting', 'cycling', 'going upstairs', 'bending', 'going downstairs', 'skating', 'scootering', 'running', 'lying']


# because a single person is also regareded as a group, activities are set to the same as actions
# JRDB-act: "Note that for singleton # groups (groups with one member), the social activity labels
# is identical to the person’s individual actions."

def jrdb_path(img_root, ann_root):
    train_seqs = [
        "tressider-2019-03-16_0",
        "svl-meeting-gates-2-2019-04-08_1",
        "svl-meeting-gates-2-2019-04-08_0",
        "stlc-111-2019-04-19_0",
        "packard-poster-session-2019-03-20_2",
        "packard-poster-session-2019-03-20_1",
        "packard-poster-session-2019-03-20_0",
        "memorial-court-2019-03-16_0",
        "jordan-hall-2019-04-22_0",
        "huang-lane-2019-02-12_0",
        "huang-basement-2019-01-25_0",
        "hewlett-packard-intersection-2019-01-24_0",
        "gates-to-clark-2019-02-28_1",
        "gates-basement-elevators-2019-01-17_1",
        "gates-159-group-meeting-2019-04-03_0",
        "forbes-cafe-2019-01-22_0",
        "cubberly-auditorium-2019-04-22_0",
        "clark-center-intersection-2019-02-28_0",
        "clark-center-2019-02-28_0",
        "bytes-cafe-2019-02-07_0",
    ]
    val_seqs = [
        "clark-center-2019-02-28_1",
        "gates-ai-lab-2019-02-08_0",
        "huang-2-2019-01-25_0",
        "meyer-green-2019-03-16_0",
        "nvidia-aud-2019-04-18_0",
        "tressider-2019-03-16_1",
        "tressider-2019-04-26_2",
    ]

    train_img_path = [img_root / f'image_{str(i*2)}' / train_seq for i in range(5) for train_seq in train_seqs]
    val_img_path = [img_root / f'image_{str(i*2)}' / val_seq for i in range(5) for val_seq in val_seqs]

    train_ann_path = [ann_root / f'{train_seq}_image{str(i*2)}.json' for train_seq in train_seqs for i in range(5)]
    val_ann_path = [ann_root / f'{val_seq}_image{str(i*2)}.json' for val_seq in val_seqs for i in range(5)]

    # train_detection_path = [ann_root / f'{train_seq}_image{str(i*2)}.json' for train_seq in train_seqs for i in range(5)]
    # val_detection_path = [ann_root / f'{val_seq}_image{str(i*2)}.json' for val_seq in val_seqs for i in range(5)]

    PATHS = {
        "train": train_ann_path,
        "val": val_ann_path
    }

    train_ann_file = PATHS['train']
    val_ann_file = PATHS['val']  # imgs and anns paths

    return train_ann_file, val_ann_file


def _parse_person_id(label_id: str, zero_based: bool = False) -> int:
    # "pedestrian:24" -> 24
    m = re.search(r":(\d+)$", label_id)
    if m is None:
        raise ValueError(f"Bad label_id format: {label_id}")
    pid = int(m.group(1))
    return pid - 1 if zero_based else pid  # start by 0

def _argmax_label_action(score_dict: dict) -> str | None:
    # {"standing":2,"holding sth":3} -> "holding sth"
    if not score_dict:
        return None
    filtered = {
        k: v for k, v in score_dict.items()
        if k in Action_names
    }

    if not filtered:
        # print('action', score_dict)
        return None

    return max(filtered.items(), key=lambda kv: kv[1])[0]
    # return max(score_dict.items(), key=lambda kv: kv[1])[0]


def _argmax_label_activity(score_dict: dict) -> str | None:
    # {"standing":2,"holding sth":3} -> "holding sth"
    if not score_dict:
        return None
    filtered = {
        k: v for k, v in score_dict.items()
        if k in Activity_names
    }

    if not filtered:
        # print('activity', score_dict)
        return None

    return max(filtered.items(), key=lambda kv: kv[1])[0]

def remap_person_ids(persons):
    """
    persons: list of dicts with 'person_id'
    return: updated persons, old_id -> new_id map
    """
    old_ids = sorted({p["person_id"] for p in persons})
    id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

    for p in persons:
        p["person_id"] = id_map[p["person_id"]]

    return persons, id_map

def remap_group_ids(groups, persons, person_id_map):
    """
    groups: list of dicts with 'group_id', 'include_id'
    """
    old_gids = sorted({g["group_id"] for g in groups})
    gid_map = {old_gid: new_gid for new_gid, old_gid in enumerate(old_gids)}  # start by 1

    for g in groups:
        g["group_id"] = gid_map[g["group_id"]]
        g["include_id"] = [person_id_map[pid] for pid in g["include_id"]]
    for p in persons:
        p["group_id"] = gid_map[p["group_id"]]

    return persons, groups, gid_map


def box_iou_xyxy(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2] (list)
    return: float IoU
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def jrdb_read_detections(detection_file):
    detections = {}
    with open(detection_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    dets = data.get("detections", {})

    for frame, objs in dets.items():
        frame_id = int(frame.split('.')[0])
        detections[frame_id] = []

        for obj in objs:
            box = obj.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(float, box)
            if x1 < 0: x1 = 0.0
            if y1 < 0: y1 = 0.0
            bbox = [x1, y1, x1 + x2, y1 + y2]
            detections[frame_id].append(bbox)

    return detections

def jrdb_read_annotations(ann_file, seq_detections):
    annotations = {}  # annotations for each frame
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", {})

    action_name_to_id = {name: i for i, name in enumerate(Action_names)}
    activity_name_to_id = {name: i for i, name in enumerate(Activity_names)}

    for frame, objs in labels.items():
        frame_id = int(frame.split('.')[0])
        annotations[frame_id] = {}
        annotations[frame_id]['groups'] = []
        annotations[frame_id]['persons'] = []

        detections = seq_detections[frame_id]

        if len(detections) == 0:
            continue

        for obj in objs:
            attr = obj.get("attributes")
            occ = attr["occlusion"]
            noeval = attr["no_eval"]
            if occ != "Fully_occluded" and noeval == False:
                box = obj.get("box", [0, 0, 0, 0])
                x1, y1, x2, y2 = map(float, box)
                if x1 < 0: x1 = 0.0
                if y1 < 0: y1 = 0.0
                bbox = [x1, y1, x1 + x2, y1 + y2]

                ious = []
                for bbox_dec in detections:
                    iou = box_iou_xyxy(bbox, bbox_dec)
                    ious.append(iou)
                iou_max = max(ious)
                if iou_max < 0.5:
                    continue

                person_id = _parse_person_id(obj.get("label_id", ""))

                action_name = _argmax_label_action(obj.get("action_label", {}) or {})
                if action_name in ('impossible', 'None', None):
                    # action_name = 'none'
                    continue
                action= action_name_to_id[action_name]

                sg = obj.get("social_group", {}) or {}  # no key -> {}, no value -> {}
                group_id = int(sg.get("cluster_ID", -1)) - 1  # start by 0

                activity_name = _argmax_label_activity(obj.get("social_activity", {}) or {})
                # print(action_name, activity_name)
                if activity_name not in Activity_names:
                    activity_name = action_name
                if activity_name == None:
                    activity = activity_name_to_id[action_name]
                else:
                    activity = activity_name_to_id[activity_name]

                if any(person.get('group_id') == group_id for person in annotations[frame_id]['persons']):
                    group = [group for group in annotations[frame_id]['groups'] if group.get('group_id') == group_id][0]
                    group['include_id'].append(person_id)
                else:
                    annotations[frame_id]['groups'].append({
                                    'group_id': group_id,
                                    'activity': activity,
                                    'include_id': [person_id]
                                })

                annotations[frame_id]['persons'].append({
                                'person_id': person_id,
                                'bbox': bbox,
                                'action': action,
                                'group_id': group_id
                            })

        persons, person_id_map = remap_person_ids(annotations[frame_id]["persons"])
        persons, groups, group_id_map = remap_group_ids(annotations[frame_id]["groups"], persons, person_id_map)
        annotations[frame_id]["persons"] = persons
        annotations[frame_id]["groups"] = groups
        annotations[frame_id]['persons'].sort(key=lambda x: x['person_id'])
        annotations[frame_id]['groups'].sort(key=lambda x: x['group_id'])

    return annotations


def jrdb_read_dataset(detection_folder, ann_files):
    data = {}
    detection = {}
    for ann_file in ann_files:
        sid = str(ann_file).split('/')[-1].split('.')[0]  # sid eg: tressider-2019-03-16_0_image0
        detection_file = detection_folder + '/' + sid + '.json'
        detection[sid] = jrdb_read_detections(detection_file)
        data[sid] = jrdb_read_annotations(ann_file, detection[sid])  # data for each seq
    return data


'''
data structure:
data: {1: {ann1}, 2: {ann2}, ... , seq: {annseq}}
ann1(seq): {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1(seq)-1(frame): {persons: [{person_id: 1, bbox: [], action:, 1, group_id: 1}, ..., {...}], 
        groups: [{group_id: 1, activity: 1, person_ids: []}, ..., {...}]}
'''


def jrdb_all_frames(anns, num_frames):
    half_left = num_frames // 2
    half_right = num_frames - half_left
    out = []
    for s in anns:
        frames = sorted(int(f) for f in anns[s].keys())

        max_f = frames[-1]

        for f in frames:
            if f % 15 != 0:     # (sid, fid) with anns (every 15 frames)
                continue
            if len(anns[s][f]["persons"]) == 0:
                continue
            if (                # filtered the first and the last anns
                f != 1
                and f != max_f
                and f + half_right <= max_f
                and f - half_left >= 1
            ):
                out.append((s, f))
    return out



class jrdb_Dataset(data.Dataset):
    def __init__(self, dataset, anns, frames, img_path, transform, num_frames=10, is_training=True):
        """
        Args:
            Characterize collective dataset based on feature maps.
        """
        self.dataset = dataset

        self.anns = anns
        self.frames = frames
        self.img_path = img_path

        self.num_frames = num_frames  # number of stacked frame features
        self.is_training = is_training

        self.transform = transform

    def __len__(self):
        return len(self.frames)  # number of frames with anns (filtered the first and thea last ones)

    def __getitem__(self, idx):
        # Load feature map

        select_frames = self.get_frames(self.frames[idx])  # 10 frames
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]
            # normal training: each training loading 10 frames
        else:
            half_left = self.num_frames // 2
            half_right = self.num_frames - half_left

            fids = range(src_fid - half_left,
                         src_fid + half_right)

            return sid, src_fid, [(sid, src_fid, fid) for fid in fids]            # normal testing: each test loading 10 frames

    def load_samples_sequence(self, select_frames):
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         device = torch.device('cpu')

        sid = select_frames[0]
        src_fid = select_frames[1]

        person_ids = []
        bboxes = []
        actions = []
        # p_group_ids = []

        group_ids = []
        activities = []
        include_ids = []

        for person in self.anns[sid][src_fid]['persons']:
            person_id = person['person_id']  # to connect the prediction and build pred matrix for iou group-level loss of group members
            person_ids.append(person_id)  # it is not in order in the tensor, following the order of group id, and some disappear
            bbox = person['bbox']  # for ROI Align and positional encoding
            bboxes.append(bbox)
            action = person['action']  # gt individual actions
            actions.append(action)
            # p_group_id = person['group_id']  # gt for person-level cross-entropy loss of group members
            # p_group_ids.append(p_group_id)

        for group in self.anns[sid][src_fid]['groups']:
            activity = group['activity']  # gt for group activity
            activities.append(activity)
            # group_id = group['group_id']
            # group_ids.append(group_id)
            include_id = group['include_id']
            include_ids.append(include_id)

        num_persons = len(person_ids)
        num_groups = len(activities)
        one_hot_matrix = np.zeros((num_persons, num_groups), dtype=float)
        # TODO: check if the order of column and the row should change
        person_to_index = {p: i for i, p in enumerate(person_ids)}

        for group, persons in enumerate(include_ids):
            for person in persons:
                one_hot_matrix[person_to_index[person], group] = 1

        imgs = []
        bbox = bboxes.copy()
        bbox = np.array(bbox, dtype=np.float64).reshape(-1, 4)
        for i, (sid, src_fid, fid) in enumerate(select_frames[2]):  # 10 frames for 1 item
            imgfolder1 = sid.split('_')[-1].replace("image", "image_")
            imgfolder2 = sid.split('_image')[0]

            if self.dataset == 'jrdb':
                img = cv2.imread(self.img_path + '/' + imgfolder1 + '/' + imgfolder2 + '/' + '/%06d.jpg' % (fid))[:, :, [2, 1, 0]]  # BGR -> RGB  # H, W, 3
            elif self.dataset == 'jrdb_group':
                SEQ = int(os.listdir(self.img_path + '/' + imgfolder1 + '/' + imgfolder2)[0].split('_')[1])
                img = cv2.imread(self.img_path + '/' + imgfolder1 + '/' + imgfolder2 + '/' + 'SEQ_%02d_%03d.jpg' % (SEQ, fid))[:, :, [2, 1, 0]]  # BGR -> RGB  # H, W, 3


            img = Image.fromarray(img)
            img, new_bboxes = self.transform(img, bbox)  # bboxes did not changed here
            imgs.append(img)

        # labels for the whole video clip (the label of the key frame)
        meta = {}
        imgs = np.stack(imgs)
        # bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, 4)
        actions = np.array(actions, dtype=np.int32)
        activities = np.array(activities, dtype=np.int32)
        one_hot_matrix = np.array(one_hot_matrix, dtype=np.int32)

        imgs = torch.from_numpy(imgs).float()
        imgs = torch.squeeze(imgs, 1)  # shape: (10, 3, H, W)
        bboxes = torch.from_numpy(new_bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        one_hot_matrix = torch.from_numpy(one_hot_matrix).int()

        meta['sid'] = sid
        meta['src_fid'] = src_fid
        meta['frame_size'] = (480, 752)

        return imgs, bboxes, actions, activities, one_hot_matrix, meta