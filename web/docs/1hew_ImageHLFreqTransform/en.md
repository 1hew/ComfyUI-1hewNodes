# Image HL Freq Transform

**Node Function:** The `Image HL Freq Transform` node implements advanced detail transfer technology, capable of transferring high-frequency information from detail images to generated images, with mask control and multiple frequency separation method support.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `generate_image` | Required | IMAGE | - | - | Generated base image |
| `detail_image` | Required | IMAGE | - | - | Detail reference image |
| `method` | - | COMBO[STRING] | igbi | rgb, hsv, igbi | Frequency separation method |
| `blur_radius` | - | FLOAT | 10.0 | 0.0-100.0 | Gaussian blur radius |
| `detail_mask` | Optional | MASK | - | - | Detail area mask, controls transfer regions |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Final image after detail transfer |
| `high_freq` | IMAGE | Blended high-frequency layer |
| `low_freq` | IMAGE | Low-frequency layer of generated image |

## Features

### Transfer Principles Details

#### RGB/HSV Mode
- **Processing flow**:
  1. Perform frequency separation on detail image to get high-frequency layer
  2. Perform frequency separation on generated image to get high and low frequency layers
  3. Blend high-frequency layers through mask
  4. Recombine final image
- **Blur radius adjustment**: Generated image uses 1.5x blur radius for better results
- **Characteristics**: Traditional method, suitable for general detail transfer needs

#### IGBI Mode (Recommended)
- **Low-frequency layer**: Gaussian blur result of the generated image
- **High-frequency layer calculation flow**:
  1. **Generated image processing**: Invert → Gaussian blur → 50% blend with blurred image → Invert again
  2. **Detail image processing**: Invert → Gaussian blur → 50% blend with blurred image → Invert again
  3. **Mask blending**: Blend the two processed results through mask
- **Final composition**: 65% high-frequency + 35% low-frequency, then apply levels adjustment (black point 83, white point 172)
- **Characteristics**: Provides the most precise and natural detail transfer effect

### Mask Control Mechanism
- **With Mask**:
  - Detail transfer only in white areas of the mask
  - Black areas preserve original details of generated image
  - Supports grayscale masks for gradient transitions
- **Without Mask**:
  - Detail transfer across the entire image
  - Automatically creates full white mask