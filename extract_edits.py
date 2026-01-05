#!/usr/bin/env python3
"""
Lightroom Catalog Edit Extractor

Extracts develop settings from a Lightroom catalog database
and exports to JSON, CSV, and Parquet formats.

Only extracts settings intended for ML training (see settings_i_want.md).
"""

import sqlite3
import json
import re
from pathlib import Path

import pandas as pd


# Default catalog path
DEFAULT_DB_PATH = Path("lightroom catalogue/lightroom catalogue.lrcat")

# Tone curve settings (arrays)
TONE_CURVES = [
    'ToneCurvePV2012',
    'ToneCurvePV2012Red',
    'ToneCurvePV2012Green',
    'ToneCurvePV2012Blue',
]

# Develop settings to extract (only those for ML training)
SETTINGS_TO_EXTRACT = [
    # Basic Panel
    'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012',
    'Whites2012', 'Blacks2012',
    
    # Presence
    'Texture', 'Clarity2012', 'Dehaze', 'Vibrance', 'Saturation',
    
    # White Balance (incremental from As Shot)
    'IncrementalTemperature', 'IncrementalTint',
    
    # HSL - Hue
    'HueAdjustmentRed', 'HueAdjustmentOrange', 'HueAdjustmentYellow',
    'HueAdjustmentGreen', 'HueAdjustmentAqua', 'HueAdjustmentBlue',
    'HueAdjustmentPurple', 'HueAdjustmentMagenta',
    
    # HSL - Saturation
    'SaturationAdjustmentRed', 'SaturationAdjustmentOrange', 'SaturationAdjustmentYellow',
    'SaturationAdjustmentGreen', 'SaturationAdjustmentAqua', 'SaturationAdjustmentBlue',
    'SaturationAdjustmentPurple', 'SaturationAdjustmentMagenta',
    
    # HSL - Luminance
    'LuminanceAdjustmentRed', 'LuminanceAdjustmentOrange', 'LuminanceAdjustmentYellow',
    'LuminanceAdjustmentGreen', 'LuminanceAdjustmentAqua', 'LuminanceAdjustmentBlue',
    'LuminanceAdjustmentPurple', 'LuminanceAdjustmentMagenta',
    
    # Color Grading
    'ColorGradeBlending',
    'ColorGradeGlobalHue', 'ColorGradeGlobalSat', 'ColorGradeGlobalLum',
    'ColorGradeShadowHue', 'ColorGradeShadowSat', 'ColorGradeShadowLum',
    'ColorGradeMidtoneHue', 'ColorGradeMidtoneSat', 'ColorGradeMidtoneLum',
    'ColorGradeHighlightHue', 'ColorGradeHighlightSat', 'ColorGradeHighlightLum',
    
    # Calibration
    'ShadowTint',
    'RedHue', 'RedSaturation',
    'GreenHue', 'GreenSaturation',
    'BlueHue', 'BlueSaturation',
]


def _extract_value(text: str, key: str):
    """Extract a single value from Lua-like settings text."""
    pattern = rf'\b{key}\s*=\s*("([^"]*)"|([-\d.]+)|(true|false))'
    match = re.search(pattern, text)
    if not match:
        return None
    
    if match.group(2) is not None:  # String
        return match.group(2)
    elif match.group(3) is not None:  # Number
        num = match.group(3)
        return float(num) if '.' in num else int(num)
    elif match.group(4) is not None:  # Boolean
        return match.group(4) == 'true'
    return None


def _extract_tone_curve(text: str, curve_name: str) -> list:
    """Extract tone curve array from settings text."""
    pattern = rf'{curve_name}\s*=\s*\{{\s*([^}}]+)\s*\}}'
    match = re.search(pattern, text)
    if not match:
        return []
    
    numbers = re.findall(r'[-\d.]+', match.group(1))
    return [float(n) if '.' in n else int(n) for n in numbers]


def extract_settings(text: str, before_text: str = None) -> dict:
    """Extract develop settings from raw settings text.
    
    For Temperature and Tint, computes incremental values (delta from As Shot).
    """
    if not text:
        return {}
    
    settings = {}
    
    # Extract scalar settings
    for key in SETTINGS_TO_EXTRACT:
        if key == 'IncrementalTemperature':
            if before_text:
                current = _extract_value(text, 'Temperature')
                before = _extract_value(before_text, 'Temperature')
                if current is not None and before is not None:
                    settings[key] = current - before
        elif key == 'IncrementalTint':
            if before_text:
                current = _extract_value(text, 'Tint')
                before = _extract_value(before_text, 'Tint')
                if current is not None and before is not None:
                    settings[key] = current - before
        else:
            value = _extract_value(text, key)
            if value is not None:
                settings[key] = value
    
    # Extract tone curves
    for curve in TONE_CURVES:
        values = _extract_tone_curve(text, curve)
        if values:
            settings[curve] = values
    
    return settings


def query_catalog(db_path: Path) -> list[dict]:
    """Query the Lightroom catalog and extract all image settings with metadata."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            i.id_local as image_id,
            f.baseName || '.' || f.extension as filename,
            fld.pathFromRoot as folder,
            rf.name as root_folder_name,
            i.captureTime,
            i.pick,
            i.rating,
            cam.value as camera_model,
            lens.value as lens,
            exif.aperture,
            exif.focalLength,
            exif.isoSpeedRating,
            exif.shutterSpeed,
            ds.text,
            bs.beforeText
        FROM Adobe_images i
        LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local
        LEFT JOIN AgLibraryFile f ON i.rootFile = f.id_local
        LEFT JOIN AgLibraryFolder fld ON f.folder = fld.id_local
        LEFT JOIN AgLibraryRootFolder rf ON fld.rootFolder = rf.id_local
        LEFT JOIN AgHarvestedExifMetadata exif ON exif.image = i.id_local
        LEFT JOIN AgInternedExifCameraModel cam ON cam.id_local = exif.cameraModelRef
        LEFT JOIN AgInternedExifLens lens ON lens.id_local = exif.lensRef
        LEFT JOIN Adobe_imageDevelopBeforeSettings bs  ON bs.developSettings = ds.id_local
        WHERE ds.text IS NOT NULL AND bs.beforeText IS NOT NULL AND i.pick >= 0
        ORDER BY i.captureTime DESC
    ''')
    
    results = []
    for row in cursor.fetchall():
        (image_id, filename, folder, root_folder_name, capture_time, pick, rating,
         camera_model, lens, aperture, focal_length, iso, shutter_speed,
         text, before_text) = row
        
        # Combine root folder, folder, and filename to get full path
        # e.g., "2025" + "/" + "2025-03-15/" + "DSC00115.ARW"
        if root_folder_name and folder:
            file_path = f"{root_folder_name}/{folder}{filename}"
        elif folder:
            file_path = f"{folder}{filename}"
        else:
            file_path = filename
        
        # Build full folder path
        full_folder = f"{root_folder_name}/{folder}" if root_folder_name and folder else folder
        
        results.append({
            'image_id': image_id,
            'filename': filename,
            'folder': full_folder,
            'file_path': file_path,
            'capture_time': capture_time,
            'pick': pick,
            'rating': rating,
            'camera_model': camera_model,
            'lens': lens,
            'aperture': aperture,
            'focal_length': focal_length,
            'iso': iso,
            'shutter_speed': shutter_speed,
            'settings': extract_settings(text, before_text),
        })
    
    conn.close()
    return results


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to a flattened DataFrame (excludes array columns)."""
    rows = []
    for r in results:
        row = {
            'image_id': r['image_id'],
            'filename': r['filename'],
            'folder': r['folder'],
            'file_path': r['file_path'],
            'capture_time': r['capture_time'],
            'rating': r['rating'],
            'pick': r['pick'],
            'camera_model': r['camera_model'],
            'lens': r['lens'],
            'aperture': r['aperture'],
            'focal_length': r['focal_length'],
            'iso': r['iso'],
            'shutter_speed': r['shutter_speed'],
        }
        for key in SETTINGS_TO_EXTRACT:
            val = r['settings'].get(key)
            row[key] = val if not isinstance(val, list) else None
        rows.append(row)
    
    return pd.DataFrame(rows)


# def export_json(results: list[dict], path: Path) -> None:
#     """Export results to JSON (includes tone curve arrays)."""
#     with open(path, 'w') as f:
#         json.dump(results, f, indent=2, default=str)


def export_csv(df: pd.DataFrame, path: Path) -> None:
    """Export DataFrame to CSV."""
    df.to_csv(path, index=False)


def export_parquet(df: pd.DataFrame, path: Path) -> None:
    """Export DataFrame to Parquet."""
    df.to_parquet(path, index=False)


def extract_edits(
    db_path: Path = DEFAULT_DB_PATH,
    output_dir: Path = Path('.'),
) -> pd.DataFrame:
    """
    Main extraction function. Queries catalog and exports to all formats.
    
    Returns the DataFrame for further processing.
    """
    results = query_catalog(db_path)
    df = to_dataframe(results)
    
    # export_json(results, output_dir / 'lightroom_edits.json')
    export_csv(df, output_dir / 'lightroom_edits.csv')
    export_parquet(df, output_dir / 'lightroom_edits.parquet')
    
    return df


if __name__ == "__main__":
    df = extract_edits()
    print(f"Extracted {len(df)} images to lightroom_edits.[json|csv|parquet]")
