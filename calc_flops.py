import argparse
import types
from collections import defaultdict

import torch
import torch.nn as nn

from models.models import GADTR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GADTR model FLOPs/MACs with dummy input using the official Cafe input setting by default."
    )

    parser.add_argument("--dataset", default="cafe", type=str)
    parser.add_argument("--image_width", default=1280, type=int)
    parser.add_argument("--image_height", default=720, type=int)
    parser.add_argument("--num_frame", default=5, type=int)
    parser.add_argument("--num_class", default=6, type=int)

    parser.add_argument("--backbone", default="resnet18", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--frozen_batch_norm", action="store_true")
    parser.add_argument("--hidden_dim", default=256, type=int)

    parser.add_argument("--num_boxes", default=14, type=int)
    parser.add_argument("--crop_size", default=5, type=int)

    parser.add_argument("--gar_nheads", default=4, type=int)
    parser.add_argument("--gar_enc_layers", default=6, type=int)
    parser.add_argument("--gar_ffn_dim", default=512, type=int)
    parser.add_argument("--position_embedding", default="sine", type=str)
    parser.add_argument("--num_group_tokens", default=12, type=int)
    parser.add_argument("--distance_threshold", default=0.2, type=float)
    parser.add_argument("--drop_rate", default=0.1, type=float)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument(
        "--use_pretrained_backbone",
        action="store_true",
        help="Use torchvision pretrained backbone weights. Disabled by default to avoid downloads.",
    )

    return parser.parse_args()


def disable_torchvision_pretrained_download(backbone_name):
    import torchvision.models as tv_models

    original_builder = getattr(tv_models, backbone_name)

    def builder_without_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_builder(*args, **kwargs)

    setattr(tv_models, backbone_name, builder_without_pretrained)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_dummy_inputs(args, device):
    images = torch.randn(
        args.batch_size,
        args.num_frame,
        3,
        args.image_height,
        args.image_width,
        device=device,
    )

    boxes = torch.zeros(args.batch_size, args.num_frame, args.num_boxes, 4, device=device)
    for i in range(args.num_boxes):
        x = 0.15 + 0.7 * ((i % 7) / 6.0)
        y = 0.25 + 0.5 * ((i // 7) / max(1, (args.num_boxes - 1) // 7))
        boxes[:, :, i, 0] = x
        boxes[:, :, i, 1] = y
        boxes[:, :, i, 2] = 0.08
        boxes[:, :, i, 3] = 0.16

    dummy_mask = torch.zeros(args.batch_size, args.num_boxes, dtype=torch.bool, device=device)
    return images, boxes, dummy_mask


class FlopCounter:
    """Counts multiply-accumulates; reported FLOPs use 1 MAC = 2 FLOPs."""

    def __init__(self):
        self.macs_by_type = defaultdict(int)
        self.macs_by_scope = defaultdict(int)
        self.handles = []
        self.patched_mha = []
        self.module_names = {}

    def add_hooks(self, model):
        self.module_names = {module: name for name, module in model.named_modules()}
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv2d_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, nn.MultiheadAttention):
                self._patch_mha_forward(module)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, original_forward in self.patched_mha:
            module.forward = original_forward
        self.patched_mha.clear()

    def _add_macs(self, module, module_type, macs):
        self.macs_by_type[module_type] += macs
        module_name = self.module_names.get(module, "")
        scope = "backbone" if module_name.startswith("backbone") else "non_backbone"
        self.macs_by_scope[scope] += macs

    def _patch_mha_forward(self, module):
        original_forward = module.forward
        counter = self

        def wrapped_forward(self_module, *args, **kwargs):
            query = kwargs.get("query", args[0] if len(args) > 0 else None)
            key = kwargs.get("key", args[1] if len(args) > 1 else None)
            if query is not None and key is not None:
                counter._count_mha(self_module, query, key)
            return original_forward(*args, **kwargs)

        module.forward = types.MethodType(wrapped_forward, module)
        self.patched_mha.append((module, original_forward))

    def _conv2d_hook(self, module, inputs, output):
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        macs = batch_size * out_channels * out_h * out_w * kernel_ops
        self._add_macs(module, "Conv2d", macs)

    def _linear_hook(self, module, inputs, output):
        macs = output.numel() * module.in_features
        self._add_macs(module, "Linear", macs)

    def _count_mha(self, module, query, key):
        if getattr(module, "batch_first", False):
            batch_size, target_len, embed_dim = query.shape
            source_len = key.shape[1]
        else:
            target_len, batch_size, embed_dim = query.shape
            source_len = key.shape[0]

        q_proj = batch_size * target_len * embed_dim * embed_dim
        k_proj = batch_size * source_len * embed_dim * embed_dim
        v_proj = batch_size * source_len * embed_dim * embed_dim
        attention_scores = batch_size * target_len * source_len * embed_dim
        attention_values = batch_size * target_len * source_len * embed_dim
        out_proj = batch_size * target_len * embed_dim * embed_dim

        macs = (
            q_proj + k_proj + v_proj + attention_scores + attention_values + out_proj
        )
        self._add_macs(module, "MultiheadAttention", macs)

    @property
    def total_macs(self):
        return sum(self.macs_by_type.values())


def fmt_number(value):
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return str(value)


def fmt_compact(value):
    if value >= 1e9:
        return f"{value / 1e9:.2f}G"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return str(value)


def print_scope_line(name, macs, total_macs):
    flops = macs * 2
    total_flops = total_macs * 2
    percent = (flops / total_flops * 100) if total_flops else 0.0
    print(
        f"{name:<24}"
        f"{macs:,} MACs ({fmt_compact(macs)}),  "
        f"{flops:,} FLOPs ({fmt_compact(flops)}), "
        f"{percent:.1f}%"
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    if not args.use_pretrained_backbone:
        disable_torchvision_pretrained_download(args.backbone)

    model = GADTR(args).to(device)
    model.eval()

    images, boxes, dummy_mask = make_dummy_inputs(args, device)

    counter = FlopCounter()
    counter.add_hooks(model)
    with torch.no_grad():
        model(images, boxes, dummy_mask)
    counter.remove_hooks()

    total_params, trainable_params = count_params(model)
    total_macs = counter.total_macs
    total_flops = total_macs * 2
    backbone_macs = counter.macs_by_scope["backbone"]
    non_backbone_macs = counter.macs_by_scope["non_backbone"]

    print("Input:")
    print(f"  images: B={args.batch_size}, T={args.num_frame}, C=3, H={args.image_height}, W={args.image_width}")
    print(f"  boxes:  B={args.batch_size}, T={args.num_frame}, N={args.num_boxes}, 4")
    print()
    print("Model:")
    print(f"  backbone={args.backbone}, hidden_dim={args.hidden_dim}, layers={args.gar_enc_layers}")
    print(f"  group_tokens={args.num_group_tokens}, heads={args.gar_nheads}")
    print()
    print("Parameters:")
    print(f"  total:     {fmt_number(total_params)}")
    print(f"  trainable: {fmt_number(trainable_params)}")
    print()
    print("MACs by module type:")
    for name, macs in sorted(counter.macs_by_type.items()):
        print(f"  {name:<20} {fmt_number(macs)} MACs")
    print()
    print("MACs/FLOPs by scope:")
    print_scope_line("backbone", backbone_macs, total_macs)
    print_scope_line("non_backbone", non_backbone_macs, total_macs)
    print()
    print(f"Total MACs:  {fmt_number(total_macs)}")
    print(f"Total FLOPs: {fmt_number(total_flops)}  (using 1 MAC = 2 FLOPs)")
    print()
    print("Note: RoIAlign, softmax, normalization, masking, indexing, and elementwise ops are not included.")


if __name__ == "__main__":
    main()
