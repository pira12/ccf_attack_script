# Image Transformation Script

This script applies multiple transformations to an input image and saves the resulting images to a specified output folder.

## Prerequisites
Make sure you have Python installed.

## Installation
1. Clone or download this repository.
2. Navigate to the project folder.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To run the script, use the following command:

```bash
python script.py --image path/to/input/image.png --output path/to/output/folder
```

### Arguments
- `--image`: Path to the input image file.
- `--output`: Path to the folder where the transformed images will be saved.

## Transformations Applied
The script applies the following transformations:
1. Embedding Additional Data (Placeholder)
2. Image Resizing
3. JPEG Compression
4. Gaussian Blur
5. Noise Addition
6. Brightness Adjustment
7. Overlay
8. Cropping
9. Rotating
10. Screenshot Simulation
11. Histogram Equalization

## Example
```bash
python script.py --image sample_image.png --output output_folder
```

All transformed images will be saved in the `output_folder` with descriptive filenames.

## License
This project is licensed under the MIT License.
