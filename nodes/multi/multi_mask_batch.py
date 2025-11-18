from comfy_api.latest import io
import torch
import torch.nn.functional as F


class MultiMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_MultiMaskBatch",
            display_name="Multi Mask Batch",
            category="1hewNodes/multi",
            inputs=[
                io.Combo.Input("fit", options=["crop", "pad", "stretch"], default="pad"),
                io.Float.Input("pad_color", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Mask.Input("mask_1"),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    async def execute(cls, fit, pad_color, **kwargs) -> io.NodeOutput:
        ordered = []
        for k in kwargs.keys():
            if k.startswith("mask_"):
                suf = k[len("mask_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        masks = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            masks.append(val.cpu())

        if not masks:
            empty = torch.zeros((0, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty)

        ref_h = int(masks[0].shape[1])
        ref_w = int(masks[0].shape[2])
        pad_value = float(pad_color)

        aligned = []
        for m in masks:
            h = int(m.shape[1])
            w = int(m.shape[2])
            if fit == "stretch":
                out = F.interpolate(
                    m.unsqueeze(1),
                    size=(ref_h, ref_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(out)
            elif fit == "crop":
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = max(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = F.interpolate(
                    m.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(
                    cls._center_crop_mask(resized, ref_h, ref_w)
                )
            else:
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = min(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = F.interpolate(
                    m.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(
                    cls._pad_mask(resized, ref_h, ref_w, pad_value)
                )

        result = (
            torch.cat(aligned, dim=0) if len(aligned) > 1 else aligned[0]
        )
        result = torch.clamp(result, min=0.0, max=1.0).to(torch.float32)
        return io.NodeOutput(result)

    @staticmethod
    def _pad_mask(mask, target_h, target_w, value):
        b, h, w = mask.shape
        out = torch.zeros(
            (b, target_h, target_w), dtype=mask.dtype, device=mask.device
        )
        out[:] = float(value)
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        h_end = min(top + h, target_h)
        w_end = min(left + w, target_w)
        out[:, top:h_end, left:w_end] = mask[:, : h_end - top, : w_end - left]
        return out

    @staticmethod
    def _center_crop_mask(mask, target_h, target_w):
        b, h, w = mask.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        top_end = min(top + target_h, h)
        left_end = min(left + target_w, w)
        cropped = mask[:, top:top_end, left:left_end]
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = MultiMaskBatch._pad_mask(cropped, target_h, target_w, 0.0)
        return cropped
