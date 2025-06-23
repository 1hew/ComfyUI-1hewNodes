# Image Crop Edge

Crop image edges - supports cropping all four sides simultaneously or setting individual crop amounts for each side.

## Parameters

- **image**: Input image
- **left_amount**: Left crop amount (0-1 for percentage, >=1 for pixels)
- **right_amount**: Right crop amount (0-1 for percentage, >=1 for pixels)
- **top_amount**: Top crop amount (0-1 for percentage, >=1 for pixels)
- **bottom_amount**: Bottom crop amount (0-1 for percentage, >=1 for pixels)
- **uniform_amount**: Uniform crop amount for all sides (overrides individual settings when > 0)
- **divisible_by**: Ensure output dimensions are divisible by this value (default: 8)

## Features

- **Flexible Input**: Supports both percentage (0-1) and pixel (>=1) values
- **Uniform Cropping**: Use uniform_amount to crop all sides equally
- **Individual Control**: Set different crop amounts for each side
- **Dimension Alignment**: Automatically adjusts to ensure output dimensions are divisible by specified value
- **Batch Processing**: Supports batch image processing

## Usage

1. Connect your image to the input
2. Set crop amounts using either:
   - **uniform_amount** for equal cropping on all sides
   - Individual **left_amount**, **right_amount**, **top_amount**, **bottom_amount** for precise control
3. Adjust **divisible_by** if you need specific dimension requirements
4. The node will output the cropped image

## Notes

- When **uniform_amount** > 0, it overrides individual side settings
- Values between 0-1 are treated as percentages of the image dimension
- Values >= 1 are treated as pixel amounts
- Output dimensions are automatically adjusted to be divisible by the specified value