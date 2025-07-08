# Image HL Freq Combine

**Node Function:** The `Image HL Freq Combine` node is used to recombine high-frequency and low-frequency image layers, supporting multiple blending modes and intensity adjustments for precise control over high and low frequency component contributions.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `high_freq` | Required | IMAGE | - | - | High-frequency detail layer image |
| `low_freq` | Required | IMAGE | - | - | Low-frequency base layer image |
| `method` | - | COMBO[STRING] | rgb | rgb, hsv, igbi | Recombination method, should match the method used for separation |
| `high_strength` | - | FLOAT | 1.0 | 0.0-2.0 | High-frequency strength adjustment, controls detail layer contribution |
| `low_strength` | - | FLOAT | 1.0 | 0.0-2.0 | Low-frequency strength adjustment, controls base layer contribution |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Recombined complete image |

## Features

### Recombination Methods Details

#### RGB Mode
- **Recombination algorithm**: Linear Light blending mode
- **Formula**: `2 * high_freq + low_freq - 1`
- **Strength adjustment**: High-frequency strength formula `(high_freq - 0.5) * strength + 0.5`
- **Characteristics**: Suitable for general image recombination with natural results

#### HSV Mode
- **Recombination algorithm**: Linear Light blending in HSV color space V channel
- **Processing flow**:
  1. Convert low-frequency image to HSV space
  2. Extract H, S, V channels
  3. Apply Linear Light blending on V channel
  4. Recombine HSV and convert back to RGB
- **Strength adjustment**: Same as RGB mode
- **Characteristics**: Preserves hue and saturation, adjusts only luminance

#### IGBI Mode
- **Recombination algorithm**: Blending + levels adjustment
- **Processing flow**:
  1. 65% high-frequency + 35% low-frequency blending
  2. Apply levels adjustment (black point 83, white point 172)
- **Strength adjustment**: Direct multiplication adjustment for high-frequency `high_freq * strength`
- **Characteristics**: Provides the most precise recombination effect