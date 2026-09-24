"""CAFE qualitative visualization matching decoding-human-association.

The renderer intentionally keeps the reference implementation's colours,
line widths, label positions, group-box padding, and output naming scheme.
"""

import colorsys
import os

import cv2
import torch
from scipy.optimize import linear_sum_assignment


ACTION_NAMES = [
    'Queueing',
    'Ordering',
    'Eating/Drinking',
    'Working/Studying',
    'Fighting',
    'TakingSelfie',
]
ACTIVITY_NAMES = ACTION_NAMES + ['Empty']


class DistinctColorGenerator:
    def __init__(self, saturation=0.7, value=0.9):
        self.index = 0
        self.s = saturation
        self.v = value

    def next(self):
        h = (self.index * 0.61803398875) % 1
        r, g, b = colorsys.hsv_to_rgb(h, self.s, self.v)
        self.index += 1
        return int(b * 255), int(g * 255), int(r * 255)


def _label_name(label, names):
    label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    if label < 0 or label >= len(names):
        return 'individual'
    return names[label]


def merge_group_bboxes(bboxes, group_ids):
    group_bboxes = []
    person_ids = []
    for gid in torch.unique(group_ids):
        group_mask = group_ids == gid
        group_person_ids = torch.where(group_mask)[0]
        group_boxes = bboxes[group_mask]
        group_bboxes.append([
            group_boxes[:, 0].min().item(),
            group_boxes[:, 1].min().item(),
            group_boxes[:, 2].max().item(),
            group_boxes[:, 3].max().item(),
        ])
        person_ids.append(group_person_ids)
    return group_bboxes, person_ids


def _clip_box(box, width, height, padding=0):
    """Convert a box to clipped integer image coordinates."""
    x1, y1, x2, y2 = map(float, box)
    return (
        max(0, min(width - 1, int(round(x1)) - padding)),
        max(0, min(height - 1, int(round(y1)) - padding)),
        max(0, min(width - 1, int(round(x2)) + padding)),
        max(0, min(height - 1, int(round(y2)) + padding)),
    )


def _rectangles_overlap(a, b, gap=3):
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_thin_text(img, text, candidates, color, font_scale, occupied):
    """Draw readable text at the first candidate that avoids other labels."""
    height, width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    placements = []
    for x, y in candidates:
        x = max(1, min(int(x), max(1, width - text_w - 2)))
        y = max(text_h + 2, min(int(y), height - baseline - 2))
        rect = (x, y - text_h, x + text_w, y + baseline)
        placements.append((x, y, rect))

    placement = next(
        (item for item in placements
         if not any(_rectangles_overlap(item[2], other) for other in occupied)),
        placements[0],
    )
    x, y, rect = placement
    cv2.putText(
        img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
    )
    occupied.append(rect)


def _member_set(member_ids):
    if isinstance(member_ids, torch.Tensor):
        member_ids = member_ids.detach().cpu().tolist()
    return {int(person_id) for person_id in member_ids}


def _matched_group_colors(pred_person_ids, gt_person_ids):
    """Assign deterministic colors and share them across matched pred/GT groups."""
    color_gen = DistinctColorGenerator(saturation=0.82, value=0.72)
    gt_colors = [color_gen.next() for _ in gt_person_ids]
    pred_colors = [color_gen.next() for _ in pred_person_ids]

    if not pred_person_ids or not gt_person_ids:
        return pred_colors, gt_colors

    pred_sets = [_member_set(members) for members in pred_person_ids]
    gt_sets = [_member_set(members) for members in gt_person_ids]
    overlap = torch.zeros((len(pred_sets), len(gt_sets)), dtype=torch.float32)
    for pred_index, pred_members in enumerate(pred_sets):
        for gt_index, gt_members in enumerate(gt_sets):
            union = pred_members | gt_members
            if union:
                overlap[pred_index, gt_index] = (
                    len(pred_members & gt_members) / len(union)
                )

    pred_indices, gt_indices = linear_sum_assignment(-overlap.numpy())
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        if overlap[pred_index, gt_index] > 0:
            pred_colors[pred_index] = gt_colors[gt_index]
    return pred_colors, gt_colors


def draw_bboxes(img, bboxes, action_labels, group_bboxes, person_ids,
                activity_labels, group_colors=None):
    """Render paper-friendly CAFE groups, including singleton individuals."""
    img_copy = img.copy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

    height, width = img_copy.shape[:2]
    scale = max(0.9, min(1.5, min(width, height) / 720.0))
    font_scale = 0.78 * scale
    line_thickness = max(1, int(round(2 * scale)))
    group_padding = max(8, int(round(14 * scale)))
    fill_alpha = 0.24

    groups = []
    if group_colors is None:
        color_gen = DistinctColorGenerator(saturation=0.82, value=0.72)
        group_colors = [color_gen.next() for _ in group_bboxes]
    for group_index, group_bbox in enumerate(group_bboxes):
        members = person_ids[group_index]
        if isinstance(members, torch.Tensor):
            members = members.detach().cpu().numpy()
        members = [int(pid) for pid in members]
        color = group_colors[group_index]
        groups.append((group_index, members, color, group_bbox))

    for _, _, color, group_bbox in groups:
        gx1, gy1, gx2, gy2 = _clip_box(
            group_bbox, width, height, group_padding
        )
        pastel = tuple(
            int(round(0.60 * channel + 0.40 * 255)) for channel in color
        )
        roi = img_copy[gy1:gy2 + 1, gx1:gx2 + 1]
        fill = roi.copy()
        fill[:] = pastel
        cv2.addWeighted(fill, fill_alpha, roi, 1.0 - fill_alpha, 0, dst=roi)

    occupied = []
    for group_index, members, color, group_bbox in groups:
        gx1, gy1, gx2, gy2 = _clip_box(
            group_bbox, width, height, group_padding
        )
        cv2.rectangle(
            img_copy, (gx1, gy1), (gx2, gy2), color, line_thickness,
            cv2.LINE_AA,
        )

        is_individual = len(members) == 1
        activity = (
            'individual' if is_individual
            else _label_name(activity_labels[group_index], ACTIVITY_NAMES)
        )
        activity_text_h = cv2.getTextSize(
            activity, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )[0][1]
        _draw_thin_text(
            img_copy,
            activity,
            [
                (gx1 + 2, gy1 - 5),
                (gx1 + 2, gy1 - activity_text_h - 10),
                (gx1 + 2, gy1 - 2 * activity_text_h - 15),
            ],
            color,
            font_scale,
            occupied,
        )

        for pid in members:
            if not 0 <= pid < len(bboxes):
                continue
            px1, py1, px2, py2 = _clip_box(bboxes[pid], width, height)
            cv2.rectangle(
                img_copy, (px1, py1), (px2, py2), color, line_thickness,
                cv2.LINE_AA,
            )
            if is_individual:
                continue
            action = _label_name(action_labels[pid], ACTION_NAMES)
            text_h = cv2.getTextSize(
                action, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )[0][1]
            _draw_thin_text(
                img_copy,
                action,
                [
                    (px1 + 1, py1 - 5),
                    (px1 + 1, py1 - text_h - 10),
                    (px1 + 1, py1 - 2 * text_h - 15),
                ],
                color,
                font_scale,
                occupied,
            )

    return img_copy


def _xywh_to_pixel_xyxy(boxes, image_width, image_height):
    boxes = boxes.detach().cpu().clone()
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((
        (x - w / 2) * image_width,
        (y - h / 2) * image_height,
        (x + w / 2) * image_width,
        (y + h / 2) * image_height,
    ), dim=-1)


def _ground_truth_groups(target, valid_actor_mask):
    """Return GT groups in reference order, including singleton outliers."""
    members = target['members'][0].detach().cpu()
    activities = target['activities'][0].detach().cpu()
    valid_indices = torch.where(valid_actor_mask.detach().cpu())[0]

    person_ids = []
    activity_labels = []
    assigned = torch.zeros(len(valid_indices), dtype=torch.bool)

    for group_idx in range(members.shape[0]):
        group_members = torch.where(members[group_idx, valid_indices] > 0)[0]
        if group_members.numel() == 0:
            continue
        person_ids.append(group_members)
        activity_labels.append(activities[group_idx])
        assigned[group_members] = True

    # The official loader stores CAFE outliers as membership=-1 instead of
    # explicit groups. The reference visualizer renders each one as a group of
    # one, after all annotated groups.
    for person_idx in torch.where(~assigned)[0]:
        person_ids.append(person_idx.reshape(1))
        activity_labels.append(torch.tensor(-2))

    return person_ids, activity_labels


@torch.no_grad()
def save_batch_visualizations(targets, infos, outputs, args):
    """Save prediction and ground-truth images for one inference batch."""
    output_dir = args.visualization_path
    gt_output_dir = output_dir + '_gt'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    pred_actions = outputs['pred_actions'].argmax(dim=-1).detach().cpu()
    pred_activities = outputs['pred_activities'].argmax(dim=-1).detach().cpu()
    memberships = outputs['membership'].detach().cpu()

    for batch_idx, (target, info) in enumerate(zip(targets, infos)):
        image_path = os.path.join(
            args.data_path,
            'cafe',
            str(info['vid']),
            str(info['sid']),
            'images',
            'frames_%s.jpg' % info['key_frame'],
        )
        key_img = cv2.imread(image_path)
        if key_img is None:
            raise FileNotFoundError('Unable to read visualization image: %s' % image_path)
        key_img = cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB)
        image_height, image_width = key_img.shape[:2]

        action_gt = target['actions'][0].detach().cpu()
        valid_actor_mask = action_gt != args.num_class + 1
        # gt_boxes contains only real actors, while actions/membership are
        # padded to args.num_boxes by the official CAFE loader.
        gt_boxes = target['gt_boxes'][0]
        num_valid_actors = int(valid_actor_mask.sum().item())
        if len(gt_boxes) != num_valid_actors:
            raise ValueError(
                'CAFE visualization actor/box mismatch: %d actors, %d boxes'
                % (num_valid_actors, len(gt_boxes))
            )
        boxes = _xywh_to_pixel_xyxy(gt_boxes, image_width, image_height)

        membership = memberships[batch_idx, :, valid_actor_mask].transpose(0, 1)
        pred_group_ids = membership.argmax(dim=-1)
        pred_group_bboxes, pred_person_ids = merge_group_bboxes(
            boxes, pred_group_ids
        )
        unique_pred_groups = torch.unique(pred_group_ids)
        pred_activity_labels = pred_activities[batch_idx, unique_pred_groups]
        pred_action_labels = pred_actions[batch_idx, valid_actor_mask]

        gt_person_ids, gt_activity_labels = _ground_truth_groups(
            target, valid_actor_mask
        )
        gt_group_ids = torch.empty(len(boxes), dtype=torch.long)
        for group_idx, group_person_ids in enumerate(gt_person_ids):
            gt_group_ids[group_person_ids] = group_idx
        gt_group_bboxes, gt_person_ids = merge_group_bboxes(boxes, gt_group_ids)

        pred_group_colors, gt_group_colors = _matched_group_colors(
            pred_person_ids, gt_person_ids
        )

        pred_img = draw_bboxes(
            key_img, boxes, pred_action_labels, pred_group_bboxes,
            pred_person_ids, pred_activity_labels, pred_group_colors,
        )
        gt_img = draw_bboxes(
            key_img, boxes, action_gt[valid_actor_mask], gt_group_bboxes,
            gt_person_ids, gt_activity_labels, gt_group_colors,
        )

        filename = 'img_seq%s_%s_frame%s.jpg' % (
            info['vid'], info['sid'], info['key_frame']
        )
        pred_path = os.path.join(output_dir, filename)
        gt_path = os.path.join(gt_output_dir, filename)
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        if not cv2.imwrite(
            pred_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR), jpeg_params
        ):
            raise OSError('Failed to write visualization: %s' % pred_path)
        if not cv2.imwrite(
            gt_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR), jpeg_params
        ):
            raise OSError('Failed to write visualization: %s' % gt_path)
