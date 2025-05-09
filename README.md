# PyLidar Leaf Scanner

A Python toolset for processing and converting laser scanner CSV files to LAZ format. This tool is specifically designed for hemispherical, hinge, and ground scan types from leaf scanners.

## Features

- Converts laser scanner CSV files to LAZ format
- Processes both first and last return points with their intensity values
- Includes range (r) and height (z) as custom LAZ parameters
- Handles different scan types (hemi, hinge, ground)
- Supports parallel processing for batch conversions

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda

### Setting up the environment

1. Clone this repository:
   ```bash
   git clone https://github.com/username/pylidar_leaf.git
   cd pylidar_leaf
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate pylidar_leaf
   ```

## Usage

### Single File Processing

To process a single CSV file:

```bash
python src/leaf_to_xyz.py data/input/scan_file.csv -o data/output/scan_file.laz
```

### Batch Processing

To process multiple files in a directory:

```bash
python scripts/process_laser_scans.py data/input/ -o data/output/
```

### Command Line Options

#### Main Conversion Tool (src/leaf_to_xyz.py)

```
python src/leaf_to_xyz.py <input_file> [options]
```

Options:
- `-o, --output`: Path to the output LAZ file (default: same as input with .laz extension)
- `-s, --sensor-height`: Height of the sensor above ground level
- `-m, --max-range`: Maximum valid range for points (default: 120)
- `-t, --transform`: Apply tilt transformation (default: True)
- `--scan-type`: Type of scan ('hemi', 'hinge', or 'ground', default: 'hemi')
- `-f, --point-format`: LAS/LAZ point format ID (default: 6)

#### Batch Processing Tool (scripts/process_laser_scans.py)

```
python scripts/process_laser_scans.py <input_path> [options]
```

Options:
- `-o, --output-dir`: Output directory (default: same as input)
- `-n, --num-workers`: Number of parallel workers (default: 10)
- `-r, --recursive`: Process directories recursively
- `-m, --max-range`: Maximum valid range for points (default: 120)
- `-s, --sensor-height`: Height of the sensor above ground level
- `-t, --transform`: Apply tilt transformation (default: True)
- `--scan-type`: Type of scan ('hemi', 'hinge', or 'ground', default: 'hemi'). 
  Also filters input files to only process files of this scan type.
- `-f, --point-format`: LAS/LAZ point format ID (default: 6)
- `-v, --verbose`: Enable verbose output

## Examples

### Basic conversion of a hemi scan

```bash
python src/leaf_to_xyz.py data/scans/ESS00354_0023_hemi_20250316-010033Z_0800_0400.csv
```

### Converting with a specific sensor height

```bash
python src/leaf_to_xyz.py data/scans/ESS00354_0023_hemi_20250316-010033Z_0800_0400.csv -s 1.5
```

### Converting with a custom output name and maximum range

```bash
python src/leaf_to_xyz.py data/scans/ESS00354_0023_hemi_20250316-010033Z_0800_0400.csv -o data/output/custom_name.laz -m 150
```

### Processing all hemi scans in a directory

```bash
python scripts/process_laser_scans.py data/scans/ --scan-type hemi -o data/output/
```

### Processing all scans recursively with multiple workers

```bash
python scripts/process_laser_scans.py data/scans/ -r -n 8 -o data/output/
```

### Processing only hinge scans with verbose output

```bash
python scripts/process_laser_scans.py data/scans/ --scan-type hinge -v -o data/output/
```

## Output Format

The output LAZ files contain the following attributes:
- Standard LAS/LAZ fields (X, Y, Z, intensity, etc.)
- Custom field `range`: Distance from scanner to point in meters
- Custom field `height`: Height (Z) value normalized to start from 0

## License

[MIT License](LICENSE)

## Acknowledgements

Based on original code by John Armston, University of Maryland.
