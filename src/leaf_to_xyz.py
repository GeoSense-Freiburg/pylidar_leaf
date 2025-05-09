#!/usr/bin/env python3
"""
leaf_to_xyz.py

Script to convert laser scanner CSV files to LAZ format.
This script processes files from a laser scanner and converts them to LAZ format
with both first and last return points.

Based on code by John Armston, University of Maryland
"""

import sys
import re
import os
import ast
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import laspy


class LaserScanFile:
    def __init__(self, filename, sensor_height=None, transform=True, 
                 max_range=120, zenith_offset=0, scan_type='hemi'):
        """
        Initialize a LaserScanFile object.
        
        Parameters:
        -----------
        filename : str
            Path to the CSV file to process
        sensor_height : float, optional
            Height of the sensor above ground level
        transform : bool, optional
            Whether to apply tilt transformation
        max_range : float, optional
            Maximum valid range for points
        zenith_offset : float, optional
            Offset to apply to zenith angles
        scan_type : str, optional
            Type of scan ('hemi', 'hinge', or 'ground')
        """
        self.filename = filename
        self.sensor_height = sensor_height
        self.transform = transform
        self.max_range = max_range
        self.zenith_offset = zenith_offset
        self.scan_type = scan_type
        
        # Try to extract metadata from filename
        pattern = re.compile(r'(\w{8})_(\d{4})_(hemi|hinge|ground)_(\d{8})-(\d{6})Z_(\d{4})_(\d{4})\.csv')
        fileinfo = pattern.search(os.path.basename(filename))
        if fileinfo:
            self.serial_number = fileinfo.group(1)
            self.scan_count = int(fileinfo.group(2))
            self.scan_type = fileinfo.group(3)
            self.datetime = datetime.strptime(f'{fileinfo.group(4)}{fileinfo.group(5)}', '%Y%m%d%H%M%S')
            self.zenith_shots = int(fileinfo.group(6))
            self.azimuth_shots = int(fileinfo.group(7))
        else:
            print(f'Note: {filename} does not match the expected LEAF scan file pattern.')
            self.serial_number = 'UNKNOWN'
            self.scan_count = 0
            self.datetime = datetime.now()
            self.zenith_shots = 0
            self.azimuth_shots = 0
        
        self.read_meta()
        self.read_data()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def read_meta(self):
        """
        Read the file header metadata.
        """
        with open(self.filename, 'r') as f:
            self.header = {}
            self.footer = {}
            in_header = True
            for line in f:
                if line.startswith('#'):
                    if 'Finished' in line:
                        lparts = line.strip().split()
                        try:
                            self.duration = float(lparts[2])
                        except (IndexError, ValueError):
                            self.duration = 0.0
                    elif 'GPS' in line:
                        lparts = line[4:].strip().split(',')
                        self.gps = lparts
                    else:
                        try:
                            lparts = line.strip().split(':')
                            if len(lparts) >= 2:
                                key = lparts[0][1:].strip()
                                val = ':'.join(lparts[1:]).strip()
                                if key in ('Batt', 'Curr', 'Lidar Temp', 'Motor Temp',
                                        'Encl. Temp', 'Encl. humidity'):
                                    val = val.split()[0]
                                try:
                                    val = ast.literal_eval(val)
                                except (SyntaxError, ValueError):
                                    pass
                                if in_header:
                                    self.header[key] = val
                                else:
                                    self.footer[key] = val
                        except Exception as e:
                            print(f"Warning: Could not parse header line: {line.strip()}")
                else:
                    in_header = False

    def read_data(self):
        """
        Read the data from the CSV file.
        """
        # Set column names and data types based on firmware version
        try:
            firm_ver = float(self.header.get('Firmware ver.', 0))
        except (ValueError, TypeError):
            firm_ver = 0
            
        if firm_ver >= 4.11:
            scan_nsteps = 2.56e4
            dtypes = {
                'sample_count': int, 
                'scan_encoder': float, 
                'rotary_encoder': float,
                'range1': float, 
                'intensity1': int, 
                'range2': float, 
                'intensity2': int, 
                'sample_time': float
            }
            col_names = list(dtypes.keys())
        else:
            scan_nsteps = 1e4
            dtypes = {
                'sample_count': int, 
                'scan_encoder': float, 
                'rotary_encoder': float,
                'range1': float, 
                'intensity1': int, 
                'range2': float, 
                'intensity2': int, 
                'sample_time': float
            }
            col_names = list(dtypes.keys())

        # Check if the file has the expected number of columns
        with open(self.filename, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    # Find first non-header line
                    parts = line.strip().split(',')
                    if len(parts) < len(col_names):
                        # If we have fewer columns than expected, adjust our expectations
                        print(f"Warning: File has fewer columns than expected ({len(parts)} vs {len(col_names)})")
                        if len(parts) == 7:  # Common case with 7 columns
                            col_names = ['sample_count', 'scan_encoder', 'rotary_encoder',
                                        'range1', 'intensity1', 'range2', 'intensity2']
                            col_names.append('sample_time')  # Add sample_time as it may be calculated later
                    break

        try:
            # Read the data with flexible parsing
            self.data = pd.read_csv(self.filename, 
                                   comment='#', 
                                   na_values=[-1.0, 654.36], 
                                   names=col_names,
                                   on_bad_lines='warn')
        except pd.errors.ParserError as e:
            print(f"Error reading CSV: {e}")
            print("Attempting to read file with more flexible settings...")
            self.data = pd.read_csv(self.filename, 
                                   comment='#',
                                   na_values=[-1.0, 654.36],
                                   header=None,
                                   on_bad_lines='skip')
            # Assign column names if possible
            if self.data.shape[1] >= len(col_names):
                self.data.columns = col_names + list(range(len(col_names), self.data.shape[1]))
            else:
                self.data.columns = col_names[:self.data.shape[1]]

        if self.data.empty:
            print("Warning: No data was read from the file.")
            return

        # Handle missing columns
        for col in dtypes.keys():
            if col not in self.data.columns:
                if col == 'intensity2' and 'range2' in self.data.columns:
                    # If we have range2 but no intensity2, create intensity2 with default values
                    self.data['intensity2'] = 100
                elif col == 'sample_time':
                    # If sample_time is missing, create it with default values
                    self.data['sample_time'] = 5.25  # Use average from example data
                else:
                    # Add missing columns as NaN
                    self.data[col] = np.nan
        
        # Try to convert columns to the correct data types
        for col, dtype in dtypes.items():
            if col in self.data.columns:
                try:
                    if dtype == int:
                        # For integer columns, first convert to float then to int to handle NaN values
                        self.data[col] = self.data[col].astype(float).fillna(0).astype(int)
                    else:
                        self.data[col] = self.data[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")

        # Remove truncated records
        if 'sample_time' in self.data.columns:
            num_short_lines = self.data.shape[0] - self.data['sample_time'].count()
            if num_short_lines > 0:
                idx = self.data['sample_time'].notna()
                self.data = self.data.loc[idx]
                msg = f'Removed {num_short_lines:d} truncated records in {self.filename}'
                print(msg)

        # Set invalid values to NaN
        for n in (1, 2):
            range_col = f'range{n}'
            intensity_col = f'intensity{n}'
            
            if range_col in self.data.columns:
                # Identify values beyond the maximum range
                mask = (self.data[range_col] > self.max_range)
                
                # Special handling for 654.36 values which indicate invalid measurements
                mask |= (self.data[range_col] == 654.36)
                
                # If intensity column exists, also filter based on intensity
                if intensity_col in self.data.columns:
                    mask |= (self.data[intensity_col] <= 0)
                
                # Set invalid ranges to NaN
                self.data.loc[mask, range_col] = np.nan

        # Calculate target count (number of valid returns)
        if 'range1' in self.data.columns and 'range2' in self.data.columns:
            self.data['target_count'] = (
                (~self.data['range1'].isna()).astype(int) + 
                (~self.data['range2'].isna()).astype(int)
            )
        else:
            self.data['target_count'] = (~self.data['range1'].isna()).astype(int)

        # Calculate datetime for each sample
        if 'sample_time' in self.data.columns:
            self.data['datetime'] = [self.datetime + timedelta(milliseconds=s)
                for s in self.data['sample_time'].cumsum()]

        # Calculate zenith and azimuth angles
        if 'scan_encoder' in self.data.columns and 'rotary_encoder' in self.data.columns:
            self.data['zenith'] = self.data['scan_encoder'] / scan_nsteps * 2 * np.pi + self.zenith_offset
            self.data['azimuth'] = self.data['rotary_encoder'] / 2e4 * 2 * np.pi

            # Apply tilt transformation if enabled
            if self.transform and 'Tilt' in self.header:
                try:
                    dx, dy, dz = (d / 1024 for d in self.header['Tilt'])
                    r, theta, phi = xyz2rza(dx, dy, dz)
                    self.data['zenith'] += theta
                    self.data['azimuth'] += phi
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not apply tilt transformation: {e}")

            # Apply scan-type specific adjustments
            if self.scan_type == 'hemi':
                idx = self.data['zenith'] < np.pi 
                self.data.loc[idx, 'azimuth'] = self.data.loc[idx, 'azimuth'] + np.pi
                
            # Normalize azimuth angles to [0, 2π]
            idx = self.data['azimuth'] > (2 * np.pi)
            self.data.loc[idx, 'azimuth'] = self.data.loc[idx, 'azimuth'] - (2 * np.pi)
            idx = self.data['azimuth'] < 0
            self.data.loc[idx, 'azimuth'] = self.data.loc[idx, 'azimuth'] + (2 * np.pi)
            
            # Normalize zenith angles to [0, π]
            self.data['zenith'] = (self.data['zenith'] - np.pi).abs()

            # Calculate XYZ coordinates for each return
            for n, name in enumerate(['range1', 'range2'], start=1):
                if name in self.data.columns:
                    x, y, z = rza2xyz(self.data[name], self.data['zenith'], self.data['azimuth'])
                    self.data[f'x{n}'] = x
                    self.data[f'y{n}'] = y
                    self.data[f'z{n}'] = z
                    if self.sensor_height is not None:
                        self.data[f'h{n}'] = z + self.sensor_height

    def save_to_laz(self, output_filename=None, point_format=6):
        """
        Save the scan data to a LAZ file.
        
        Parameters:
        -----------
        output_filename : str, optional
            Path to the output LAZ file. If not provided, the input filename is used with .laz extension.
        point_format : int, optional
            Point format ID for the LAZ file. Default is 6, which includes RGB colors.
        """
        if output_filename is None:
            # Use the input filename with .laz extension
            output_filename = os.path.splitext(self.filename)[0] + '.laz'
        
        # Determine the number of points for first and last returns
        num_first_returns = (~self.data['range1'].isna()).sum()
        num_last_returns = (~self.data['range2'].isna()).sum()
        total_points = num_first_returns + num_last_returns
        
        if total_points == 0:
            print(f"Error: No valid points found in {self.filename}")
            return False
        
        # Create a new LAS header
        header = laspy.LasHeader(point_format=point_format, version="1.4")
        header.scales = [0.001, 0.001, 0.001]  # Set coordinate scaling factors
        
        # Add extra dimensions for range and height
        extra_dims = [
            laspy.ExtraBytesParams(name="range", type=np.float32, description="Range (distance) in meters"),
            laspy.ExtraBytesParams(name="height", type=np.float32, description="Height (Z coordinate) in meters")
        ]
        for dim in extra_dims:
            header.add_extra_dim(dim)
        
        # Create a new LAS file
        las = laspy.LasData(header)
        
        # Create points arrays for X, Y, Z, intensity, etc.
        x_values = np.zeros(total_points, dtype=np.float64)
        y_values = np.zeros(total_points, dtype=np.float64)
        z_values = np.zeros(total_points, dtype=np.float64)
        intensity_values = np.zeros(total_points, dtype=np.uint16)
        return_num_values = np.zeros(total_points, dtype=np.uint8)
        range_values = np.zeros(total_points, dtype=np.float32)
        height_values = np.zeros(total_points, dtype=np.float32)
        gps_time_values = np.zeros(total_points, dtype=np.float64)
        
        point_index = 0
        
        # Process first returns
        if num_first_returns > 0:
            valid_first = ~self.data['range1'].isna()
            first_data = self.data[valid_first]
            
            # Fill X, Y, Z arrays
            x_values[point_index:point_index+num_first_returns] = first_data['x1'].values
            y_values[point_index:point_index+num_first_returns] = first_data['y1'].values
            z_values[point_index:point_index+num_first_returns] = first_data['z1'].values
            
            # Set intensity (with a better scaling algorithm)
            if 'intensity1' in first_data.columns:
                # Scale intensities using a logarithmic curve to better differentiate values
                # This will spread out the lower values more and compress the higher values
                intensities = first_data['intensity1'].values
                # Ensure no zeros (replace with 1)
                intensities = np.maximum(intensities, 1)
                # Apply log scaling: log10(1 + intensity) / log10(101) * 65535
                # This maps 1->0, 10->25335, 50->47115, 100->65535
                log_scaled = np.log10(1 + intensities) / np.log10(101) * 65535
                intensity_values[point_index:point_index+num_first_returns] = log_scaled.astype(np.uint16)
            
            # Set return number, number of returns
            return_num_values[point_index:point_index+num_first_returns] = 1
            
            # Set range values (from range1)
            range_values[point_index:point_index+num_first_returns] = first_data['range1'].values
            
            # Set height values (z coordinate) - store raw z values for now
            height_values[point_index:point_index+num_first_returns] = first_data['z1'].values
            
            # Optional: Set GPS time if available
            if 'datetime' in first_data.columns:
                # Convert datetime to GPS time (seconds since GPS epoch)
                epoch = datetime(1980, 1, 6)  # GPS epoch
                gps_times = np.array([(dt - epoch).total_seconds() for dt in first_data['datetime']])
                gps_time_values[point_index:point_index+num_first_returns] = gps_times
            
            # Update point index
            point_index += num_first_returns
        
        # Process last returns
        if num_last_returns > 0:
            valid_last = ~self.data['range2'].isna()
            last_data = self.data[valid_last]
            
            # Fill X, Y, Z arrays
            x_values[point_index:point_index+num_last_returns] = last_data['x2'].values
            y_values[point_index:point_index+num_last_returns] = last_data['y2'].values
            z_values[point_index:point_index+num_last_returns] = last_data['z2'].values
            
            # Set intensity (with a better scaling algorithm)
            if 'intensity2' in last_data.columns:
                # Scale intensities using a logarithmic curve to better differentiate values
                intensities = last_data['intensity2'].values
                # Ensure no zeros (replace with 1)
                intensities = np.maximum(intensities, 1)
                # Apply log scaling: log10(1 + intensity) / log10(101) * 65535
                log_scaled = np.log10(1 + intensities) / np.log10(101) * 65535
                intensity_values[point_index:point_index+num_last_returns] = log_scaled.astype(np.uint16)
            
            # Set return number
            return_num_values[point_index:point_index+num_last_returns] = last_data['target_count'].values
            
            # Set range values (from range2)
            range_values[point_index:point_index+num_last_returns] = last_data['range2'].values
            
            # Set height values (z coordinate) - store raw z values for now
            height_values[point_index:point_index+num_last_returns] = last_data['z2'].values
            
            # Optional: Set GPS time if available
            if 'datetime' in last_data.columns:
                # Convert datetime to GPS time (seconds since GPS epoch)
                epoch = datetime(1980, 1, 6)  # GPS epoch
                gps_times = np.array([(dt - epoch).total_seconds() for dt in last_data['datetime']])
                gps_time_values[point_index:point_index+num_last_returns] = gps_times
        
        # Now, create a new point record with the correct size
        las.x = x_values
        las.y = y_values
        las.z = z_values
        las.intensity = intensity_values
        las.return_number = return_num_values
        las.gps_time = gps_time_values
        
        # Normalize height values to start from 0
        if height_values.size > 0:
            min_height = height_values.min()
            if min_height < 0:
                height_values = height_values - min_height  # Shift all values up by min_height
        
        # Add the custom fields: range and height (normalized to start from 0)
        las.range = range_values
        las.height = height_values
        
        # Set all classifications to 0 (never classified)
        las.classification = np.zeros(total_points, dtype=np.uint8)
        
        # Set number_of_returns to 0 (since we can't remove it from the format)
        las.number_of_returns = np.ones(total_points, dtype=np.uint8)
        
        # Set header metadata
        las.header.generating_software = f"leaf_to_xyz.py v1.0"
        
        # Write the LAZ file
        try:
            las.write(output_filename)
            print(f"Successfully wrote {total_points} points to {output_filename}")
            return True
        except Exception as e:
            print(f"Error writing LAZ file: {e}")
            return False


def rza2xyz(r, theta, phi):
    """
    Calculate xyz coordinates from the spherical data
    Right-hand coordinate system
    
    Parameters:
    -----------
    r : array-like
        Range/radius values
    theta : array-like
        Zenith angles (in radians)
    phi : array-like
        Azimuth angles (in radians)
        
    Returns:
    --------
    tuple
        (x, y, z) coordinates
    """
    x = r * np.sin(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)

    return x, y, z


def xyz2rza(x, y, z):
    """
    Calculate spherical coordinates from the xyz data
    
    Parameters:
    -----------
    x, y, z : float or array-like
        Cartesian coordinates
        
    Returns:
    --------
    tuple
        (r, theta, phi) spherical coordinates
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Handle potential division by zero
    if np.isscalar(r):
        theta = 0 if r == 0 else np.arccos(z / r)
    else:
        theta = np.zeros_like(r)
        nonzero = r > 0
        theta[nonzero] = np.arccos(z[nonzero] / r[nonzero])
    
    phi = np.arctan2(x, y)
    
    # Normalize phi to [0, 2π]
    if np.isscalar(phi):
        if phi < 0:
            phi += 2*np.pi
        if phi > 2*np.pi:
            phi -= 2*np.pi
    else:
        phi[phi < 0] += 2*np.pi
        phi[phi > 2*np.pi] -= 2*np.pi

    return r, theta, phi


def process_scan_file(input_file, output_file=None, sensor_height=None, 
                     max_range=120, transform=True, scan_type='hemi', point_format=6):
    """
    Process a laser scan file and convert it to LAZ format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to the output LAZ file
    sensor_height : float, optional
        Height of the sensor above ground
    max_range : float, optional
        Maximum valid range for points
    transform : bool, optional
        Whether to apply tilt transformation
    scan_type : str, optional
        Type of scan ('hemi', 'hinge', or 'ground')
    point_format : int, optional
        Point format ID for the LAZ file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create scanner object
        scanner = LaserScanFile(
            filename=input_file,
            sensor_height=sensor_height,
            transform=transform,
            max_range=max_range,
            scan_type=scan_type
        )
        
        # Save to LAZ file
        success = scanner.save_to_laz(output_file, point_format=point_format)
        return success
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert laser scanner CSV files to LAZ format.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-o", "--output", help="Path to the output LAZ file")
    parser.add_argument("-s", "--sensor-height", type=float, help="Height of the sensor above ground level")
    parser.add_argument("-m", "--max-range", type=float, default=120, help="Maximum valid range for points")
    parser.add_argument("-t", "--transform", action="store_true", default=True, help="Apply tilt transformation")
    parser.add_argument("--scan-type", choices=["hemi", "hinge", "ground"], default="hemi", 
                       help="Type of scan (hemi, hinge, or ground)")
    parser.add_argument("-f", "--point-format", type=int, default=6, 
                       help="LAS/LAZ point format ID (default: 6)")
    
    args = parser.parse_args()
    
    # Process file
    success = process_scan_file(
        input_file=args.input_file,
        output_file=args.output,
        sensor_height=args.sensor_height,
        max_range=args.max_range,
        transform=args.transform,
        scan_type=args.scan_type,
        point_format=args.point_format
    )
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()