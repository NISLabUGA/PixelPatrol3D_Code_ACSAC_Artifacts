#!/usr/bin/env python3
"""
PP3D Data Download Script
Downloads the required datasets for ACSAC artifact evaluation
"""

import os
import sys
import hashlib
import zipfile
import shutil
import requests
from pathlib import Path
from typing import Dict, List
import argparse

# Configuration
DATA_URL = "https://pp3d-data.sdkhomelab.com"
DATA_DIR = Path("artifacts/pp3d_data")
REQUIRED_FILES = ["l1o.zip", "test.zip", "train.zip"]

# Expected checksums from README
CHECKSUMS = {
    "train.zip": "360c9ef325e44cb11ae0dd55baaf0983",
    "test.zip": "446cd0b764f939e50b1694f43cc7e560",
    "l1o.zip": "ccd0f7378b54cb64455a4378b2fe3847"
}

# File sizes (in GB)
FILE_SIZES = {
    "train.zip": {"compressed": 46, "uncompressed": 52},
    "test.zip": {"compressed": 1, "uncompressed": 1},
    "l1o.zip": {"compressed": 171, "uncompressed": 217}
}

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_status(message: str):
    """Print info message in blue"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message: str):
    """Print success message in green"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message: str):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message: str):
    """Print error message in red"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def check_disk_space() -> bool:
    """Check if there's sufficient disk space for the download"""
    print_status("Checking available disk space...")
    
    # Required space in GB (compressed + uncompressed for safety)
    required_space_gb = 300
    
    try:
        # Get available space
        statvfs = os.statvfs('.')
        available_bytes = statvfs.f_frsize * statvfs.f_bavail
        available_gb = available_bytes / (1024**3)
        
        print_status(f"Available disk space: {available_gb:.1f} GB")
        print_status(f"Required disk space: {required_space_gb} GB")
        
        if available_gb < required_space_gb:
            print_error(f"Insufficient disk space. Need at least {required_space_gb} GB free.")
            print_error("Please free up disk space and try again.")
            return False
        
        print_success("Sufficient disk space available.")
        return True
        
    except Exception as e:
        print_warning(f"Could not check disk space: {e}")
        return True  # Continue anyway

def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(filename: str, force: bool = False) -> bool:
    """Download a file with progress bar"""
    url = f"{DATA_URL}/{filename}"
    output_path = DATA_DIR / filename
    
    print_status(f"Downloading {filename}...")
    
    # Check if file already exists
    if output_path.exists() and not force:
        print_warning(f"{filename} already exists. Use --force to re-download.")
        return True
    
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB)", end='', flush=True)
        
        print()  # New line after progress
        print_success(f"Downloaded {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error downloading {filename}: {e}")
        return False

def verify_checksum(filename: str) -> bool:
    """Verify file checksum"""
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        print_error(f"{filename} not found for verification")
        return False
    
    print_status(f"Verifying {filename}...")
    
    try:
        actual_checksum = calculate_md5(file_path)
        expected_checksum = CHECKSUMS[filename]
        
        if actual_checksum == expected_checksum:
            print_success(f"{filename} checksum verified")
            return True
        else:
            print_error(f"{filename} checksum mismatch!")
            print_error(f"Expected: {expected_checksum}")
            print_error(f"Actual:   {actual_checksum}")
            return False
            
    except Exception as e:
        print_error(f"Error verifying {filename}: {e}")
        return False

def extract_file(filename: str) -> bool:
    """Extract a zip file"""
    zip_path = DATA_DIR / filename
    
    if not zip_path.exists():
        print_error(f"{filename} not found for extraction")
        return False
    
    print_status(f"Extracting {filename}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print_success(f"Extracted {filename}")
        return True
        
    except zipfile.BadZipFile:
        print_error(f"{filename} is not a valid zip file")
        return False
    except Exception as e:
        print_error(f"Error extracting {filename}: {e}")
        return False

def display_data_info():
    """Display information about the dataset"""
    print("\n" + "="*50)
    print("PP3D Dataset Information")
    print("="*50)
    print("\nDataset Sizes (approximate):")
    
    total_compressed = 0
    total_uncompressed = 0
    
    for filename in REQUIRED_FILES:
        sizes = FILE_SIZES[filename]
        compressed = sizes["compressed"]
        uncompressed = sizes["uncompressed"]
        total_compressed += compressed
        total_uncompressed += uncompressed
        
        print(f"  • {filename:12} {compressed:3d} GB compressed → {uncompressed:3d} GB uncompressed")
    
    print(f"\nTotal: ~{total_compressed} GB compressed → ~{total_uncompressed} GB uncompressed")
    
    print("\nThis download includes:")
    print("  • Training data for RQ1, RQ4, RQ5")
    print("  • Test data for all research questions")
    print("  • Leave-one-out data for RQ2, RQ3")
    
    print("\nNote: The download may take several hours depending on")
    print("your internet connection speed.")
    print("="*50 + "\n")

def confirm_download() -> bool:
    """Ask user for confirmation"""
    try:
        response = input("Do you want to proceed with the download? (y/N): ").strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Download PP3D dataset for ACSAC artifact evaluation")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing files")
    parser.add_argument("--no-extract", action="store_true", help="Download only, do not extract")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files")
    parser.add_argument("--cleanup", action="store_true", help="Delete zip files after extraction")
    args = parser.parse_args()
    
    print("PP3D Dataset Download Script")
    print("="*30)
    
    # If only verifying, skip download
    if args.verify_only:
        print_status("Verifying existing files...")
        all_verified = True
        for filename in REQUIRED_FILES:
            if not verify_checksum(filename):
                all_verified = False
        
        if all_verified:
            print_success("All files verified successfully!")
        else:
            print_error("Some files failed verification.")
            sys.exit(1)
        return
    
    # Display data information
    display_data_info()
    
    # Ask for confirmation
    if not confirm_download():
        print_status("Download cancelled by user.")
        return
    
    # Check disk space
    if not check_disk_space():
        sys.exit(1)
    
    # Download files
    print_status("Starting download of required dataset files...")
    download_success = True
    
    for filename in REQUIRED_FILES:
        if not download_file(filename, args.force):
            download_success = False
            break
    
    if not download_success:
        print_error("Download failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Verify checksums
    print_status("Verifying file integrity...")
    verification_success = True
    
    for filename in REQUIRED_FILES:
        if not verify_checksum(filename):
            verification_success = False
    
    if not verification_success:
        print_error("File verification failed. Please re-download the files.")
        sys.exit(1)
    
    # Extract files
    if not args.no_extract:
        print_status("Extracting downloaded files...")
        extraction_success = True
        
        for filename in REQUIRED_FILES:
            if not extract_file(filename):
                extraction_success = False
        
        if not extraction_success:
            print_error("Extraction failed.")
            sys.exit(1)
        
        # Clean up zip files if requested
        if args.cleanup:
            for filename in REQUIRED_FILES:
                zip_path = DATA_DIR / filename
                if zip_path.exists():
                    zip_path.unlink()
                    print_status(f"Deleted {filename}")
    
    print_success("Dataset download completed successfully!")
    print_status(f"Data is now available in: {DATA_DIR}")
    
    # Display final directory structure
    print("\n" + print_status("Final directory structure:"))
    try:
        for item in sorted(DATA_DIR.iterdir()):
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                size_mb = item.stat().st_size / (1024**2)
                print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    except Exception:
        print(f"  Contents of {DATA_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
