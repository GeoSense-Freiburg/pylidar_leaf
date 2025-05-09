#!/usr/bin/env python3
"""
process_laser_scans.py

Wrapper script for leaf_to_xyz.py that can process a single file or an entire directory
of CSV files, with multiprocessing capability.

Usage:
    python process_laser_scans.py <input_path> [options]
    
    <input_path> can be a single CSV file or a directory containing CSV files
    
Options:
    -o, --output-dir: Output directory (default: same as input)
    -n, --num-workers: Number of parallel workers (default: 10)
    -r, --recursive: Process directories recursively
    -m, --max-range: Maximum valid range for points (default: 120)
    -s, --sensor-height: Height of the sensor above ground level
    -t, --transform: Apply tilt transformation (default: True)
    --scan-type: Type of scan ('hemi', 'hinge', or 'ground', default: 'hemi'). 
                Also filters input files to only process files of this scan type.
    -f, --point-format: LAS/LAZ point format ID (default: 6)
    -v, --verbose: Enable verbose output
"""

import os
import sys
import glob
import argparse
import multiprocessing
from functools import partial
from pathlib import Path
import time

# Add src directory to the path so we can import leaf_to_xyz
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from leaf_to_xyz import process_scan_file


def process_file(file_path, output_dir=None, verbose=False, **kwargs):
    """
    Process a single laser scan file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file to process
    output_dir : str, optional
        Directory where the output LAZ file will be saved
    verbose : bool, optional
        Whether to print verbose output
    **kwargs : dict
        Additional arguments to pass to process_scan_file
        
    Returns:
    --------
    tuple
        (file_path, success, processing_time)
    """
    start_time = time.time()
    
    # Determine output file path
    if output_dir:
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '.laz'))
    else:
        output_file = None  # Will use the default (same directory as input)
    
    if verbose:
        print(f"Processing file: {file_path}")
        print(f"Output file: {output_file}")
        print(f"Parameters: {kwargs}")
    
    # Process the file
    try:
        success = process_scan_file(file_path, output_file, **kwargs)
        processing_time = time.time() - start_time
        return file_path, success, processing_time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, False, time.time() - start_time


def find_csv_files(input_path, recursive=False, scan_type=None):
    """
    Find all CSV files in the input path that match the scan type.
    
    Parameters:
    -----------
    input_path : str
        Path to a file or directory
    recursive : bool, optional
        Whether to search for files recursively in subdirectories
    scan_type : str, optional
        Filter for scan type ('hemi', 'hinge', or 'ground')
        
    Returns:
    --------
    list
        List of paths to CSV files
    """
    if os.path.isfile(input_path):
        # For single file processing, check if it matches the scan_type
        if input_path.lower().endswith('.csv'):
            if scan_type and scan_type in ['hemi', 'hinge', 'ground']:
                # Check if scan_type is in the filename (like "_hemi_" or "_hinge_")
                scan_pattern = f"_{scan_type}_"
                if scan_pattern in os.path.basename(input_path).lower():
                    return [input_path]
                else:
                    print(f"Skipping {input_path}: scan type does not match '{scan_type}'")
                    return []
            else:
                # If no scan_type filter is specified, process the file
                return [input_path]
        else:
            return []
    
    # If input_path is a directory
    all_csv_files = []
    if recursive:
        pattern = os.path.join(input_path, '**', '*.csv')
        all_csv_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_path, '*.csv')
        all_csv_files = glob.glob(pattern)
    
    # Filter by scan type if specified
    if scan_type and scan_type in ['hemi', 'hinge', 'ground']:
        scan_pattern = f"_{scan_type}_"
        filtered_files = [file for file in all_csv_files if scan_pattern in os.path.basename(file).lower()]
        
        skipped = len(all_csv_files) - len(filtered_files)
        if skipped > 0:
            print(f"Skipping {skipped} files that don't match scan type '{scan_type}'")
        
        return filtered_files
    else:
        return all_csv_files


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process laser scanner CSV files in parallel.")
    parser.add_argument("input_path", help="Path to a CSV file or a directory containing CSV files")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("-n", "--num-workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("-m", "--max-range", type=float, default=120, help="Maximum valid range for points (default: 120)")
    parser.add_argument("-s", "--sensor-height", type=float, help="Height of the sensor above ground level")
    parser.add_argument("-t", "--transform", action="store_true", default=True, help="Apply tilt transformation (default: True)")
    parser.add_argument("--scan-type", choices=["hemi", "hinge", "ground"], default="hemi", 
                       help="Type of scan ('hemi', 'hinge', or 'ground', default: 'hemi'). Also filters input files to only process files of this scan type.")
    parser.add_argument("-f", "--point-format", type=int, default=6, 
                       help="LAS/LAZ point format ID (default: 6)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Find CSV files to process
    csv_files = find_csv_files(args.input_path, args.recursive, args.scan_type)
    
    if not csv_files:
        print(f"No CSV files found in {args.input_path}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    # Create output directory if specified and doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    
    # Prepare processing function with fixed arguments
    process_func = partial(
        process_file,
        output_dir=args.output_dir,
        verbose=args.verbose,
        sensor_height=args.sensor_height,
        max_range=args.max_range,
        transform=args.transform,
        scan_type=args.scan_type,
        point_format=args.point_format
    )
    
    # Use the minimum of available CPUs, requested workers, and number of files
    num_workers = min(multiprocessing.cpu_count(), args.num_workers, len(csv_files))
    
    # Process files in parallel
    total_start_time = time.time()
    
    if num_workers > 1 and len(csv_files) > 1:
        print(f"Processing with {num_workers} workers...")
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(process_func, csv_files)
    else:
        print("Processing in single-threaded mode...")
        results = [process_func(file) for file in csv_files]
    
    # Report results
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    total_time = time.time() - total_start_time
    avg_time = sum(r[2] for r in results) / len(results) if results else 0
    
    print("\nProcessing summary:")
    print(f"Total files: {len(csv_files)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time:.2f} seconds")
    
    if num_workers > 1:
        speedup = avg_time * len(csv_files) / total_time if total_time > 0 else 0
        print(f"Parallel speedup: {speedup:.2f}x")
    
    if successful:
        print("\nSuccessfully processed files:")
        for file_path, _, proc_time in successful:
            print(f"- {file_path} ({proc_time:.2f} seconds)")
    
    if failed:
        print("\nFailed files:")
        for file_path, _, _ in failed:
            print(f"- {file_path}")
    
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
