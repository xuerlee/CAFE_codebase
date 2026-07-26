
from types import SimpleNamespace
from evaluation import cafe_eval

pred_path = "/home/jiqqi/code/decoding-human-association/runs_cafe/28_new_new_cafe_hidden256_enc6dec6_12queries_lossratio223_grpnllcost_bceloss_place_noflip_aligned_sepgrploss_nogradclip/pred_group_test_place_score_29.txt"
gt_path = "/media/jiqqi/OS/dataset/Cafe_Dataset/evaluation/gt_tracks.txt"
labelmap_path = "label_map/group_action_list.pbtxt"

args = SimpleNamespace(
    groundtruth=open(gt_path, "r"),
    labelmap=open(labelmap_path, "r"),
    eval_type="gt_base",
)

metrics = cafe_eval.GAD_Evaluation(args)
with open(pred_path, "r") as detections:
    result = metrics.evaluate(detections)

print(result)
print("group mAP at 1.0:", result["group_mAP_1.0"])
print("group mAP at 0.5:", result["group_mAP_0.5"])
print("outlier mIoU:", result["outlier_mIoU"])

args.groundtruth.close()
args.labelmap.close()
