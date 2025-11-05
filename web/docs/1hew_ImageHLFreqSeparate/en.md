# Image HL Freq Separate

**Node Function:** The `Image HL Freq Separate` node implements advanced frequency separation technology, supporting RGB, HSV, and IGBI separation methods. It can separate images into high-frequency detail layers and low-frequency base layers.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Image to perform frequency separation on |
| `method` | - | COMBO[STRING] | rgb | rgb, hsv, igbi | Separation method: rgb(RGB space), hsv(HSV space), igbi(Inverted Gaussian Blur Invert) |
| `blur_radius` | - | FLOAT | 10.0 | 0.0-100.0 | Gaussian blur radius, controls the frequency range of separation |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `high_freq` | IMAGE | High-frequency detail layer image |
| `low_freq` | IMAGE | Low-frequency base layer image |
| `combine` | IMAGE | Recombined complete image |

## Features

### Separation Methods Details

#### RGB Mode
- **Low-frequency layer**: Direct Gaussian blur on the original image
- **High-frequency layer**: Calculated based on grayscale information using formula `(grayscale - blurred_grayscale) / 255 + 0.5`
- **Recombination**: Uses Linear Light blending mode `2 * high_freq + low_freq - 1`
- **Characteristics**: Suitable for general image processing with high computational efficiency

#### HSV Mode
- **Low-frequency layer**: Gaussian blur on V channel while preserving H and S channels
- **High-frequency layer**: V channel difference calculation using formula `(V_channel - blurred_V_channel) / 255 + 0.5`
- **Recombination**: Linear Light blending in HSV space on V channel
- **Characteristics**: Preserves hue and saturation, processes only luminance information

#### IGBI Mode (Inverted Gaussian Blur Invert)
- **Low-frequency layer**: Direct Gaussian blur on the original image
- **High-frequency layer processing flow**:
  1. Image inversion (255 - original)
  2. Original image Gaussian blur
  3. 50% blend of inverted image and blurred image
  4. Invert again to get high-frequency layer
- **Recombination**: 65% high-frequency + 35% low-frequency, then apply levels adjustment (black point 83, white point 172)
- **Characteristics**: Provides the most precise detail separation effect