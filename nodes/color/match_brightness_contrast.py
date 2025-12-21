import torch
from comfy_api.latest import io


class MatchBrightnessContrast(io.ComfyNode):
    """
    Match Brightness & Contrast
    
    Adjusts the brightness and contrast of the source_image to match the reference_image.
    Can optionally use only the edge area for statistics calculation to ignore central content changes.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MatchBrightnessContrast",
            display_name="Match Brightness Contrast",
            category="1hewNodes/color",
            inputs=[
                io.Image.Input("source_image"),
                io.Image.Input("reference_image"),
                io.Float.Input("edge_amount", default=0.2, min=0.0, max=8192.0, step=0.01, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def _calculate_margin(cls, amount: float, height: int, width: int) -> int:
        """
        Calculate actual margin in pixels.
        amount <= 0: 0 (Use full image)
        0 < amount < 1.0: Percentage of short side
        amount >= 1.0: Pixel count
        """
        if amount <= 0:
            return 0
        
        # Percentage mode
        if amount < 1.0:
            base = min(height, width)
            return int(base * amount)
            
        # Pixel mode
        return int(amount)

    @classmethod
    def _get_stats(cls, image_tensor: torch.Tensor, margin: int):
        """
        Calculate mean and std of the image or its edges.
        image_tensor: [H, W, C]
        """
        if margin <= 0:
            mean = torch.mean(image_tensor, dim=(0, 1))
            std = torch.std(image_tensor, dim=(0, 1))
            return mean, std
            
        h, w, c = image_tensor.shape
        
        # If margin covers the whole image (overlap in center), use full image
        if margin * 2 >= h or margin * 2 >= w:
             mean = torch.mean(image_tensor, dim=(0, 1))
             std = torch.std(image_tensor, dim=(0, 1))
             return mean, std

        # Extract edges (Top, Bottom, Left Middle, Right Middle)
        top = image_tensor[:margin, :, :]
        bottom = image_tensor[h-margin:, :, :]
        
        # Side strips excluding corners to avoid double counting pixels
        left = image_tensor[margin:h-margin, :margin, :]
        right = image_tensor[margin:h-margin, w-margin:, :]
        
        # Concatenate all edge pixels
        edges = torch.cat([
            top.reshape(-1, c),
            bottom.reshape(-1, c),
            left.reshape(-1, c),
            right.reshape(-1, c)
        ], dim=0)
        
        mean = torch.mean(edges, dim=0)
        std = torch.std(edges, dim=0)
        return mean, std

    @classmethod
    async def execute(cls, source_image, reference_image, edge_amount):
        # source_image, reference_image: [B, H, W, C]
        
        src_batch = source_image.shape[0]
        ref_batch = reference_image.shape[0]
        
        res_images = []
        
        for i in range(src_batch):
            src_img = source_image[i]
            
            # Get corresponding reference image (handle batch mismatch)
            ref_idx = i % ref_batch
            ref_img = reference_image[ref_idx]
            
            # Calculate stats for source
            h_src, w_src = src_img.shape[:2]
            margin_src = cls._calculate_margin(edge_amount, h_src, w_src)
            mu_src, std_src = cls._get_stats(src_img, margin_src)
            
            # Calculate stats for reference
            h_ref, w_ref = ref_img.shape[:2]
            margin_ref = cls._calculate_margin(edge_amount, h_ref, w_ref)
            mu_ref, std_ref = cls._get_stats(ref_img, margin_ref)
            
            # Avoid division by zero
            std_src = torch.where(std_src < 1e-6, torch.ones_like(std_src), std_src)
            
            # Color transfer: (x - mu_src) * (std_ref / std_src) + mu_ref
            res = (src_img - mu_src) * (std_ref / std_src) + mu_ref
            
            res_images.append(res)
            
        result = torch.stack(res_images)
        result = torch.clamp(result, 0.0, 1.0)
        
        return io.NodeOutput(result)
