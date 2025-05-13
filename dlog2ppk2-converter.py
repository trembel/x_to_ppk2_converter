#!/usr/bin/env python

"""
dlog2ppk2-converter.py
A python script to convert DLog current measurements into the PPK2 format used by the Nordic Power Profiler application.

Copyright (C) 2025 Silvano Cortesi

This work is licensed under the terms of the MIT license.  For a copy, see the
included LICENSE file or <https://opensource.org/licenses/MIT>.
---------------------------------
"""

import struct
import json
import zipfile
import argparse
import sys
import time
import io
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timezone

# Try importing lzma for optional .xz support
try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False

PPK2_FORMAT_VERSION = 2
FRAME_SIZE_BYTES = 6  # 4 bytes current (float32 microamps) + 2 bytes bits (uint16)
DEFAULT_DIGITAL_BITS = 0xAAAA  # Default value when no digital data is present

# Folding Buffer for minimap generation (small map for navigation in PPK GUI)
class MiniFoldingBuffer:
    def __init__(self, max_elements=10000):
        self.max_elements = max_elements
        self.reset()

    def reset(self):
        self.num_times_to_fold = 1
        self.last_element_fold_count = 0
        # Initialize with empty slots, matching JS Array(size) behavior
        self.data = {
            "length": 0,
            "min": [None] * self.max_elements,
            "max": [None] * self.max_elements,
        }

    def _add_default(self, timestamp_us):
        idx = self.data["length"]
        if idx < self.max_elements:
            self.data["min"][idx] = {"x": timestamp_us, "y": float('inf')}
            self.data["max"][idx] = {"x": timestamp_us, "y": float('-inf')}
            self.data["length"] += 1
        else:
            # This case should be handled by folding before exceeding max_elements
             print("Warning: Attempted to add default beyond max_elements", file=sys.stderr)


    def _fold(self):
        new_length = self.data["length"] // 2
        for i in range(new_length):
            idx1 = i * 2
            idx2 = i * 2 + 1

            # Average timestamps
            avg_x = (self.data["min"][idx1]["x"] + self.data["min"][idx2]["x"]) / 2

            # Combine min/max points
            min_y = min(self.data["min"][idx1]["y"], self.data["min"][idx2]["y"])
            max_y = max(self.data["max"][idx1]["y"], self.data["max"][idx2]["y"])

            # Update the first half of the array
            self.data["min"][i] = {"x": avg_x, "y": min_y}
            self.data["max"][i] = {"x": avg_x, "y": max_y}

        # Clear the second half (optional, mimics JS array resizing effect implicitly)
        for i in range(new_length, self.data["length"]):
             self.data["min"][i] = None
             self.data["max"][i] = None

        self.data["length"] = new_length
        self.num_times_to_fold *= 2


    def add_data(self, value_micro_amps, timestamp_sec):
        timestamp_us = timestamp_sec * 1_000_000
        value_nA = value_micro_amps * 1_000 # Convert microampere to nanoampere for minimap

        # Apply floor value
        # This is for the minimap data, not session.raw
        if value_nA < 200:
            value_nA = 200

        if self.last_element_fold_count == 0:
            if self.data["length"] == self.max_elements:
                self._fold()
            self._add_default(timestamp_us)

        self.last_element_fold_count += 1
        current_idx = self.data["length"] - 1

        # Update min point
        min_point = self.data["min"][current_idx]
        alpha = 1.0 / self.last_element_fold_count # Weight for timestamp average
        min_point["x"] = timestamp_us * alpha + min_point["x"] * (1 - alpha)
        if not math.isnan(value_nA):
             min_point["y"] = min(value_nA, min_point["y"])

        # Update max point
        max_point = self.data["max"][current_idx]
        # Timestamp averaging is the same
        max_point["x"] = timestamp_us * alpha + max_point["x"] * (1 - alpha)
        if not math.isnan(value_nA):
            max_point["y"] = max(value_nA, max_point["y"])

        if self.last_element_fold_count == self.num_times_to_fold:
            self.last_element_fold_count = 0

    def get_state(self):
        # Return only the valid portion of the arrays
        valid_min = [d for d in self.data["min"][:self.data["length"]] if d is not None]
        valid_max = [d for d in self.data["max"][:self.data["length"]] if d is not None]

        # Ensure lengths match after potential None filtering (shouldn't happen with proper logic)
        if len(valid_min) != self.data["length"] or len(valid_max) != self.data["length"]:
             print(f"Warning: Mismatch in valid min/max lengths ({len(valid_min)}, {len(valid_max)}) vs data.length ({self.data['length']})", file=sys.stderr)


        return {
            "lastElementFoldCount": self.last_element_fold_count,
            "data": {
                "length": self.data["length"],
                "min": valid_min,
                "max": valid_max,
            },
            "maxNumberOfElements": self.max_elements,
            "numberOfTimesToFold": self.num_times_to_fold,
        }

# --- DLog Parsing Logic ---
class DLogChannelInfo:
    """Helper class to store parsed channel info"""
    def __init__(self, id, slot, smu, unit, data_index=-1): # data_index optional now
        self.id = id
        self.slot = slot
        self.smu = smu
        self.unit = unit
        self.data_index = data_index # Index in the raw numpy data array if loaded

    def __repr__(self):
        # Provide a cleaner representation for listing
        return f"ID={self.id:<3} Slot={self.slot:<2} Unit='{self.unit}'   Model='{self.smu}'"

def _open_dlog_file(filename):
    """Internal helper to open dlog file, handling optional compression."""
    is_xz = filename.lower().endswith(".xz")
    if is_xz:
        if not LZMA_AVAILABLE:
            print("Error: .xz file detected but 'lzma' module not found or failed to import.", file=sys.stderr)
            print("       Install with 'pip install pyliblzma' or ensure lzma is available.", file=sys.stderr)
            return None
        try:
            print("    Detected .xz compression, using lzma.")
            return lzma.open(filename, "rb")
        except Exception as e:
            print(f"Error opening compressed file '{filename}': {e}", file=sys.stderr)
            return None
    else:
        try:
            return open(filename, "rb")
        except FileNotFoundError:
             print(f"Error: Input DLog file not found: {filename}", file=sys.stderr)
             return None
        except Exception as e:
             print(f"Error opening file '{filename}': {e}", file=sys.stderr)
             return None

def parse_dlog_header(filename):
    """Parses only the XML header of a DLog file to extract channel info."""
    print(f"  Parsing DLog header: {filename}")
    lines = []
    xml_header_str = ""

    f = _open_dlog_file(filename)
    if f is None:
        return None # Error message already printed

    try:
        with f: # Ensure file is closed
            # Read XML part line by line until </dlog>
            while True:
                try:
                    line_bytes = f.readline()
                    if not line_bytes: # End of file reached unexpectedly
                        print(f"Error: End of file reached before finding '</dlog>' tag in {filename}", file=sys.stderr)
                        return None
                    # Decode carefully, ignore errors for robustness
                    line = line_bytes.decode('utf-8', errors='ignore')
                    lines.append(line)
                    if "</dlog>" in line:
                        break
                except UnicodeDecodeError as e:
                     print(f"Warning: Unicode decode error reading line: {e}. Trying to continue.", file=sys.stderr)
                     lines.append(str(line_bytes)) # Append raw representation
                     if "</dlog>" in str(line_bytes): # Check raw bytes too
                         break
                except EOFError:
                    print(f"Error: Unexpected end of file while reading header in {filename}", file=sys.stderr)
                    return None

            xml_header_str = "".join(lines)

        # --- Parse XML Header ---
        # Handle potentially problematic tags
        xml_header_str = xml_header_str.replace("1ua>", "X1ua>")
        xml_header_str = xml_header_str.replace("2ua>", "X2ua>")
        try:
            dlog_xml = ET.fromstring(xml_header_str)
        except ET.ParseError as e:
            print(f"Error: Failed to parse XML header: {e}", file=sys.stderr)
            return None

        all_channels_info = []
        parsed_channels_count = 0
        for channel_xml in dlog_xml.findall("channel"):
            try:
                channel_id = int(channel_xml.get("id"))
                model_elem = channel_xml.find("ident/model")
                model = model_elem.text if model_elem is not None else "Unknown"
                slot_elem = channel_xml.find("ident/slot")
                slot_num_str = slot_elem.text if slot_elem is not None else "0"
                slot_num = int(slot_num_str) if slot_num_str.isdigit() else 0 # Handle non-numeric/missing slot

                sense_curr_elem = channel_xml.find("sense_curr")
                sense_volt_elem = channel_xml.find("sense_volt")
                sense_curr = sense_curr_elem.text == "1" if sense_curr_elem is not None else False
                sense_volt = sense_volt_elem.text == "1" if sense_volt_elem is not None else False

                # Add channels based on what's sensed
                if sense_volt:
                    info = DLogChannelInfo(channel_id, slot_num, model, "V")
                    all_channels_info.append(info)
                    parsed_channels_count += 1
                if sense_curr:
                    info = DLogChannelInfo(channel_id, slot_num, model, "A")
                    all_channels_info.append(info)
                    parsed_channels_count += 1
            except (ValueError, TypeError, AttributeError) as e:
                 print(f"Warning: Skipping channel due to parsing error in XML: {e}. XML fragment:\n{ET.tostring(channel_xml, encoding='unicode')}", file=sys.stderr)


        if parsed_channels_count == 0:
             print(f"Warning: No channels found or parsed successfully in XML header of {filename}", file=sys.stderr)
             # Return empty list instead of None if XML parsed but no channels found
             return []

        print(f"    Found {len(all_channels_info)} measurement streams (V/A) across channels.")
        return all_channels_info

    except Exception as e:
        print(f"Error parsing DLog header for '{filename}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def load_dlog_data(filename, target_channel_id):
    """
    Loads data from a .dlog file, focusing on extracting metadata and the
    target current channel data.
    Returns: (sample_interval_sec, samples_per_second, np_data_array, target_channel_index, all_channels_info) or None on error
    """
    print(f"  Loading DLog file (full): {filename}")
    lines = []
    line = ""
    raw_data = None
    xml_header_str = ""

    f = _open_dlog_file(filename)
    if f is None:
        return None # Error message printed by helper

    try:
        with f: # Ensure file is closed
            # Read XML part line by line until </dlog>
            while True:
                try:
                    line_bytes = f.readline()
                    if not line_bytes:
                        print(f"Error: End of file reached before finding '</dlog>' tag in {filename}", file=sys.stderr)
                        return None
                    line = line_bytes.decode('utf-8', errors='ignore')
                    lines.append(line)
                    if "</dlog>" in line:
                        break
                except UnicodeDecodeError as e:
                     print(f"Warning: Unicode decode error reading line: {e}. Trying to continue.", file=sys.stderr)
                     lines.append(str(line_bytes))
                     if "</dlog>" in str(line_bytes):
                         break
                except EOFError:
                    print(f"Error: Unexpected end of file while reading header in {filename}", file=sys.stderr)
                    return None

            xml_header_str = "".join(lines)

            # Read the rest as binary data
            raw_data = f.read()
            print(f"    Read {len(xml_header_str)} bytes of XML header.")
            print(f"    Read {len(raw_data)} bytes of binary data.")

        if not raw_data:
            print(f"Error: No binary data found in {filename} after XML header.", file=sys.stderr)
            return None

        # --- Parse XML Header ---
        xml_header_str = xml_header_str.replace("1ua>", "X1ua>")
        xml_header_str = xml_header_str.replace("2ua>", "X2ua>")
        try:
            dlog_xml = ET.fromstring(xml_header_str)
        except ET.ParseError as e:
            print(f"Error: Failed to parse XML header: {e}", file=sys.stderr)
            return None

        all_channels_info = []
        parsed_channels = [] # List of channels contributing to data streams
        data_idx_counter = 0
        for channel_xml in dlog_xml.findall("channel"):
            try:
                channel_id = int(channel_xml.get("id"))
                model_elem = channel_xml.find("ident/model")
                model = model_elem.text if model_elem is not None else "Unknown"
                slot_elem = channel_xml.find("ident/slot")
                slot_num_str = slot_elem.text if slot_elem is not None else "0"
                slot_num = int(slot_num_str) if slot_num_str.isdigit() else 0

                sense_curr_elem = channel_xml.find("sense_curr")
                sense_volt_elem = channel_xml.find("sense_volt")
                sense_curr = sense_curr_elem.text == "1" if sense_curr_elem is not None else False
                sense_volt = sense_volt_elem.text == "1" if sense_volt_elem is not None else False

                # Store info for all V/A streams found for metadata/lookup
                current_channel_added_to_data = False
                if sense_volt:
                    info = DLogChannelInfo(channel_id, slot_num, model, "V", data_idx_counter)
                    all_channels_info.append(info)
                    parsed_channels.append(info) # This channel contributes a data stream
                    data_idx_counter += 1
                    current_channel_added_to_data = True
                if sense_curr:
                    info = DLogChannelInfo(channel_id, slot_num, model, "A", data_idx_counter)
                    all_channels_info.append(info)
                    parsed_channels.append(info) # This channel contributes a data stream
                    data_idx_counter += 1
                    current_channel_added_to_data = True

                # If neither V nor A sensed for this channel block
                if not current_channel_added_to_data:
                    print(f"Warning: Channel ID {channel_id} found in XML but senses neither V nor A.", file=sys.stderr)

            except (ValueError, TypeError, AttributeError) as e:
                 print(f"Warning: Skipping channel during data load due to parsing error in XML: {e}. XML fragment:\n{ET.tostring(channel_xml, encoding='unicode')}", file=sys.stderr)

        num_parsed_channels = len(parsed_channels) # Number of actual V/A streams expected
        if num_parsed_channels == 0:
             print(f"Error: No channels with sensed V or A found/parsed in XML header of {filename}", file=sys.stderr)
             return None

        # Get interval (sample period) and sense_minmax flag
        frame_elem = dlog_xml.find("frame")
        if frame_elem is None:
             print(f"Error: Could not find <frame> element in XML header.", file=sys.stderr)
             return None
        tint_elem = frame_elem.find("tint")
        minmax_elem = frame_elem.find("sense_minmax")
        if tint_elem is None or tint_elem.text is None:
             print(f"Error: Could not find <tint> (sample interval) in XML header.", file=sys.stderr)
             return None
        if minmax_elem is None or minmax_elem.text is None:
             print(f"Warning: Could not find <sense_minmax> flag in XML header, assuming disabled.", file=sys.stderr)
             sense_minmax = False
        else:
             sense_minmax = minmax_elem.text == "1"

        try:
            sample_interval_sec = float(tint_elem.text)
            if sample_interval_sec <= 0:
                 print(f"Error: Invalid sample interval found in XML: {sample_interval_sec}", file=sys.stderr)
                 return None
            samples_per_second = round(1.0 / sample_interval_sec) # Round to nearest Hz
        except ValueError:
            print(f"Error: Invalid format for <tint> value: {tint_elem.text}", file=sys.stderr)
            return None

        print(f"    Sample Interval (tint): {sample_interval_sec * 1e6:.3f} us")
        print(f"    Determined Sample Rate: {samples_per_second} Hz")
        print(f"    Sense Min/Max Enabled: {sense_minmax}")

        # Adjust number of data streams if min/max is enabled
        num_data_streams = num_parsed_channels
        if sense_minmax:
            num_data_streams *= 3
            print(f"    Expecting {num_data_streams} data streams ({num_parsed_channels} channels * 3 [avg,min,max])")
        else:
             print(f"    Expecting {num_data_streams} data streams ({num_parsed_channels} channels)")


        # --- Parse Binary Data ---
        expected_bytes_per_sample = 4 * num_data_streams # float32 = 4 bytes
        if expected_bytes_per_sample == 0:
            print(f"Error: Calculated zero bytes per sample. Check channel parsing.", file=sys.stderr)
            return None
        num_samples = len(raw_data) // expected_bytes_per_sample

        if len(raw_data) % expected_bytes_per_sample != 0:
             print(f"Warning: Binary data size ({len(raw_data)}) is not an exact multiple of expected bytes per sample ({expected_bytes_per_sample}). Truncating.", file=sys.stderr)
             raw_data = raw_data[:num_samples * expected_bytes_per_sample] # Use only complete samples

        if num_samples == 0:
            print(f"Error: Calculated number of samples is zero based on binary data size and number of channels.", file=sys.stderr)
            return None

        print(f"    Calculated number of samples: {num_samples}")
        observed_duration = num_samples * sample_interval_sec
        print(f"    Observed data duration: {observed_duration:.3f} s")

        # Reshape the data: DLog uses Big Endian ('>f')
        np_data_flat = np.frombuffer(raw_data, dtype='>f')

        if np_data_flat.size != num_samples * num_data_streams:
             print(f"Error: Unexpected numpy array size after parsing binary data. Expected {num_samples * num_data_streams}, got {np_data_flat.size}", file=sys.stderr)
             return None

        np_data_reshaped = np_data_flat.reshape((num_samples, num_data_streams)).T # Transpose to get (streams, samples)
        print(f"    Successfully parsed binary data into numpy array with shape: {np_data_reshaped.shape}")

        # --- Find Target Current Channel Index ---
        target_channel_info = None
        target_original_data_index = -1 # The index BEFORE sense_minmax multiplication
        for info in parsed_channels: # Search within channels that contribute data
             if info.id == target_channel_id and info.unit == 'A':
                  target_channel_info = info
                  target_original_data_index = info.data_index
                  break

        if target_channel_info is None:
            print(f"Error: Could not find a current ('A') channel with ID {target_channel_id} among the sensed channels.", file=sys.stderr)
            print(f"Available sensed channels (contributing to data):")
            if parsed_channels:
                for info in parsed_channels:
                    print(f"  ID={info.id:<3} Slot={info.slot:<2} Unit='{info.unit}' Index={info.data_index} Model='{info.smu}'")
            else:
                print("  None.")
            # Also print all found channels for context
            print(f"\nAll V/A streams found in header (metadata):")
            if all_channels_info:
                for info in all_channels_info:
                     print(f"  {info}") # Use the standard repr here
            else:
                 print("  None.")
            return None

        # Determine the actual index in the np_data_reshaped array
        target_data_index = target_original_data_index
        if sense_minmax:
            # In min/max mode, the order is usually: Avg, Min, Max for each original stream
            target_data_index *= 3
            print(f"    Adjusted target data index for min/max mode: {target_original_data_index} -> {target_data_index} (using avg/current stream)")

        if target_data_index >= np_data_reshaped.shape[0]:
             print(f"Error: Calculated target data index ({target_data_index}) is out of bounds for the parsed data array shape {np_data_reshaped.shape}.", file=sys.stderr)
             return None

        print(f"    Using data from channel: ID={target_channel_info.id} Slot={target_channel_info.slot} Unit='{target_channel_info.unit}'")
        print(f"    Data index in numpy array: {target_data_index}")

        # Return all_channels_info as well, might be useful context
        return sample_interval_sec, samples_per_second, np_data_reshaped, target_data_index, all_channels_info

    except FileNotFoundError: # Should be caught by _open_dlog_file, but keep for safety
        print(f"Error: Input DLog file not found: {filename}", file=sys.stderr)
        return None
    except ET.ParseError as e:
        print(f"Error parsing DLog XML header: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing DLog file '{filename}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


# --- Main Conversion Function ---
def convert_dlog_to_ppk2(dlog_path, start_time_str, target_channel_id, output_path):
    """
    Converts a Keysight DLog file to PPK2 format using data from a specific channel.
    """
    print(f"Starting conversion:")
    print(f"  Input DLog: {dlog_path}")
    print(f"  Target Current Channel ID: {target_channel_id}")
    print(f"  Output PPK2: {output_path}")

    # 1. Validate Start Time and Convert to Milliseconds Epoch (reuse from csv2ppk2)
    try:
        # Assume ISO 8601 format, try parsing with and without timezone awareness
        try:
            start_dt = datetime.fromisoformat(start_time_str)
        except ValueError:
             # Try adding a 'Z' for UTC if timezone is missing
             if not start_time_str.endswith('Z') and '+' not in start_time_str and '-' not in start_time_str[10:]: # Avoid hyphens in date part
                  start_dt = datetime.fromisoformat(start_time_str + 'Z')
             else:
                  raise

        # If timezone naive, assume local timezone and convert to UTC
        if start_dt.tzinfo is None or start_dt.tzinfo.utcoffset(start_dt) is None:
            print("Warning: Start time has no timezone info, assuming local time.", file=sys.stderr)
            start_dt = start_dt.astimezone() # Convert to local timezone aware
            start_dt_utc = start_dt.astimezone(timezone.utc) # Then convert to UTC
        else:
            start_dt_utc = start_dt.astimezone(timezone.utc)

        start_time_ms_epoch = start_dt_utc.timestamp() * 1000.0
        print(f"  Parsed Start Time (UTC): {start_dt_utc.isoformat(timespec='milliseconds')}")
    except ValueError as e:
        print(f"Error: Invalid start time format: {start_time_str}", file=sys.stderr)
        print(f"Please use ISO 8601 format (e.g., '2023-10-27T10:00:00Z' or '2023-10-27T12:00:00+02:00'). Details: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load DLog data
    load_result = load_dlog_data(dlog_path, target_channel_id)
    if load_result is None:
        print("Exiting due to DLog loading error.", file=sys.stderr)
        sys.exit(1) # Error message already printed by load_dlog_data

    sample_interval_sec, samples_per_second, dlog_data_np, target_idx, all_channels_info = load_result
    target_current_data_amps = dlog_data_np[target_idx] # Get the specific current channel data (already in Amperes)

    # Find the specific channel info for metadata
    target_channel_metadata = next((ch for ch in all_channels_info if ch.id == target_channel_id and ch.unit == 'A'), None)

    # 3. Process Data and Prepare for PPK2
    session_data = io.BytesIO()
    minimap_buffer = MiniFoldingBuffer()
    start_processing_time = time.time()

    print("  Processing DLog samples...")
    num_samples = len(target_current_data_amps)
    skipped_samples = 0

    for i in range(num_samples):
        current_value_amps = target_current_data_amps[i]

        # Check for invalid values often seen in dlog files
        # Allow slightly wider range than pure physical reality, but catch obvious errors
        if not np.isfinite(current_value_amps) or abs(current_value_amps) > 1e7: # Check NaN, Inf, very large values
            if skipped_samples < 10: # Report first few skips
                 print(f"Warning: Skipping invalid data value {current_value_amps} at sample index {i}. Replacing with 0.", file=sys.stderr)
            elif skipped_samples == 10:
                 print(f"Warning: Supressing further invalid data value warnings...", file=sys.stderr)
            current_value_amps = 0.0 # Replace with 0
            skipped_samples += 1

        # Convert current to microamps for processing/storage
        current_value_micro_amps = current_value_amps * 1_000_000

        # Add data to minimap buffer (relative timestamp from start)
        relative_timestamp_sec = i * sample_interval_sec
        minimap_buffer.add_data(current_value_micro_amps, relative_timestamp_sec)

        # Pack data for session.raw (float 32-bit little-endian current in microamps, unsigned 16-bit little-endian bits)
        # IMPORTANT: PPK2 uses Little Endian ('<'), DLog uses Big Endian ('>')
        try:
            packed_current = struct.pack('<f', current_value_micro_amps)
            packed_bits = struct.pack('<H', DEFAULT_DIGITAL_BITS)
            session_data.write(packed_current)
            session_data.write(packed_bits)
        except struct.error as e:
             print(f"Error packing data at sample {i} (current_uA={current_value_micro_amps}): {e}. Skipping sample.", file=sys.stderr)
             skipped_samples += 1
             continue # Skip writing this sample if packing fails


        if (i + 1) % 50000 == 0: # Print progress
            print(f"    Processed {i + 1}/{num_samples} samples...")

    if skipped_samples > 0:
        print(f"  Finished processing. Skipped {skipped_samples} invalid/problematic samples.", file=sys.stderr)

    total_bytes_written = session_data.tell()
    total_samples_written = total_bytes_written // FRAME_SIZE_BYTES
    print(f"  Finished processing {num_samples} DLog samples. Wrote {total_samples_written} samples to session buffer.")
    if total_samples_written == 0:
        print(f"Error: No valid data samples were processed or written.", file=sys.stderr)
        sys.exit(1)

    # 4. Create PPK2 (ZIP Archive)
    try:
        print(f"  Creating PPK2 file: {output_path}")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            # Create metadata.json
            metadata_content = {
                "metadata": {
                    "samplesPerSecond": samples_per_second,
                    "startSystemTime": start_time_ms_epoch, # Absolute start time
                    "createdBy": "dlog2ppk2-converter.py",
                    # Add dlog specific info
                    "sourceFileInfo": {
                         "filename": os.path.basename(dlog_path),
                         "type": "Keysight DLog",
                         "targetChannelId": target_channel_id,
                         "targetChannelSlot": target_channel_metadata.slot if target_channel_metadata else "Unknown",
                         "targetChannelModel": target_channel_metadata.smu if target_channel_metadata else "Unknown",
                    }
                },
                "formatVersion": PPK2_FORMAT_VERSION
            }
            zf.writestr('metadata.json', json.dumps(metadata_content, indent=None, separators=(',', ':'))) # Compact JSON

            # Write session.raw
            session_data.seek(0) # Rewind buffer
            zf.writestr('session.raw', session_data.read())

            # Create minimap.raw
            minimap_content = minimap_buffer.get_state()
            # Use compact JSON encoding for minimap
            zf.writestr('minimap.raw', json.dumps(minimap_content, indent=None, separators=(',', ':')))

    except Exception as e:
        print(f"Error creating PPK2 file: {e}", file=sys.stderr)
        # Clean up partially created file if error occurs
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass # Ignore cleanup errors
        sys.exit(1)

    end_processing_time = time.time()
    duration = end_processing_time - start_processing_time
    print(f"  PPK2 file created successfully.")
    print(f"  Processing time: {duration:.2f} seconds")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Keysight DLog data to Nordic PPK2 format, or list available channels.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # Input file is always required
    parser.add_argument('--dlog-input', required=True, help='Path to the input DLog file (.dlog or .dlog.xz).')

    # Action to list channels
    parser.add_argument('--list-channels', action='store_true', help='List available channels (ID, Slot, Unit, Model) present in the DLog file and exit.')

    # Arguments required only for conversion (not for listing)
    parser.add_argument('--start-time',
                        help="System start time of the recording in ISO 8601 format (e.g., '2023-10-27T15:30:00Z'). Assumed local time if no timezone provided. If no time provided, current system time is taken.")
    parser.add_argument('--channel-id', type=int,
                        help='Required for conversion: The ID of the channel containing the current (A) measurement to use.')
    parser.add_argument('--output',
                        help='Required for conversion: Path for the output PPK2 file (e.g., my_capture.ppk2).')

    args = parser.parse_args()

    # --- Action: List Channels ---
    if args.list_channels:
        print(f"Listing channels for: {args.dlog_input}")
        channel_list = parse_dlog_header(args.dlog_input)

        if channel_list is None:
            print("Failed to parse DLog header.", file=sys.stderr)
            sys.exit(1)
        elif not channel_list:
            print("No channels found in the DLog header.")
        else:
            print("\nAvailable measurement streams found in header:")
            print("-" * 40)
            # Sort by ID then Unit for consistent display
            channel_list.sort(key=lambda ch: (ch.id, ch.unit))
            for info in channel_list:
                print(f"  {info}")
            print("-" * 40)
            print("Note: Use the 'ID' of a channel with Unit='A' for the --channel-id argument during conversion.")
        sys.exit(0)

    # --- Action: Convert ---
    else:
        # Check if required arguments for conversion are provided
        missing_args = []
        if not args.start_time:
            args.start_time = datetime.now().isoformat()
        if args.channel_id is None: # Check for None because 0 could be a valid ID
            missing_args.append("--channel-id")
        if not args.output:
            missing_args.append("--output")

        if missing_args:
            parser.error(f"The following arguments are required for conversion (when not using --list-channels): {', '.join(missing_args)}")

        # Basic validation for output filename
        if not args.output.lower().endswith('.ppk2'):
             print("Warning: Output filename does not end with .ppk2", file=sys.stderr)

        # Proceed with conversion
        convert_dlog_to_ppk2(
            args.dlog_input,
            args.start_time,
            args.channel_id,
            args.output
        )
