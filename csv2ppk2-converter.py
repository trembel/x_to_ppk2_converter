import csv
import struct
import json
import zipfile
import argparse
import sys
import time
import io
import math
import os
from datetime import datetime, timezone

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

# --- Helper function to find column index ---
def find_column_index(header, column_spec, column_description):
    """
    Finds the 0-based index of a column given a header and a specification (name or 0-based index).
    """
    try:
        # Try interpreting as 0-based index
        col_index_0_based = int(column_spec)
        if 0 <= col_index_0_based < len(header):
            print(f"  Using 0-based index {col_index_0_based} for {column_description} column.")
            return col_index_0_based
        else:
            print(f"Error: {column_description} column index '{col_index_0_based}' is out of range (0-{len(header)-1}).", file=sys.stderr)
            sys.exit(1)
    except ValueError:
        # Interpret as column name (case-insensitive matching)
        col_spec_lower = column_spec.lower().strip()
        for i, col_name in enumerate(header):
            if col_name.lower().strip() == col_spec_lower:
                print(f"  Found {column_description} column by name: '{header[i]}' (index {i}).")
                return i
        print(f"Error: Could not find {column_description} column with name '{column_spec}'.", file=sys.stderr)
        print(f"Available columns: {header}", file=sys.stderr)
        sys.exit(1)

# --- Main Conversion Function ---
def convert_csv_to_ppk2(csv_path, start_time_str, timestamp_col_spec, current_col_spec, output_path):
    """
    Converts a CSV file with specified timestamp and current columns to PPK2 format.
    """
    print(f"Starting conversion:")
    print(f"  Input CSV: {csv_path}")
    print(f"  Timestamp Column Spec: '{timestamp_col_spec}' (expected in seconds)")
    print(f"  Current Column Spec: '{current_col_spec}' (expected in ampere)")
    print(f"  Output PPK2: {output_path}")

    # 1. Validate Start Time and Convert to Milliseconds Epoch
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
        print(f"  Parsed Start Time (UTC): {start_dt_utc.isoformat()}")
    except ValueError as e:
        print(f"Error: Invalid start time format: {start_time_str}", file=sys.stderr)
        print(f"Please use ISO 8601 format (e.g., '2023-10-27T10:00:00Z' or '2023-10-27T12:00:00+02:00'). Details: {e}", file=sys.stderr)
        sys.exit(1)

    samples_per_second = None
    timestamp_col_index = -1
    current_col_index = -1
    session_data = io.BytesIO()
    minimap_buffer = MiniFoldingBuffer()
    row_count = 0
    first_timestamp_sec = None
    start_processing_time = time.time()

    # 2. Process CSV
    try:
        with open(csv_path, 'r', newline='') as infile:
            reader = csv.reader(infile)

            # Read header
            try:
                header = next(reader)
                row_count += 1
                # Clean header names (remove leading/trailing whitespace)
                header = [h.strip() for h in header]
            except StopIteration:
                print(f"Error: CSV file '{csv_path}' is empty.", file=sys.stderr)
                sys.exit(1)
            if not header:
                print(f"Error: CSV file '{csv_path}' contains an empty header row.", file=sys.stderr)
                sys.exit(1)

            print(f"  CSV Header: {header}")

            # Find column indices using the helper function
            timestamp_col_index = find_column_index(header, timestamp_col_spec, "timestamp")
            current_col_index = find_column_index(header, current_col_spec, "current")

            if timestamp_col_index == current_col_index:
                print(f"Error: Timestamp and Current columns cannot be the same (index {timestamp_col_index}).", file=sys.stderr)
                sys.exit(1)

            # Read data rows and determine sample rate
            last_timestamp_sec = None
            sample_deltas = []

            print("  Processing CSV rows...")
            for row in reader:
                row_count += 1
                if not row or len(row) <= max(timestamp_col_index, current_col_index):
                    print(f"Warning: Skipping row {row_count} due to insufficient columns or empty row. Row data: {row}", file=sys.stderr)
                    continue

                try:
                    # Parse timestamp (seconds) and current (ampere)
                    current_timestamp_sec = float(row[timestamp_col_index])
                    current_value_amps = float(row[current_col_index])

                    # Convert current to microamps for processing/storage
                    current_value_micro_amps = current_value_amps * 1_000_000

                    if first_timestamp_sec is None:
                         first_timestamp_sec = current_timestamp_sec
                         print(f"    First data timestamp: {first_timestamp_sec:.6f} s")

                    # Calculate sample rate based on first few samples
                    if last_timestamp_sec is not None and len(sample_deltas) < 100: # Check ~100 samples for rate stability
                        delta = current_timestamp_sec - last_timestamp_sec
                        if delta > 1e-9: # Avoid division by zero or negative delta
                             sample_deltas.append(delta)
                        elif delta <= 0:
                             print(f"Warning: Non-increasing timestamp detected at row {row_count}: {current_timestamp_sec} <= {last_timestamp_sec}", file=sys.stderr)


                    # Determine samples_per_second after enough deltas
                    if samples_per_second is None and len(sample_deltas) >= 10:
                        avg_delta = sum(sample_deltas) / len(sample_deltas)
                        # Check variance (optional but good)
                        variance = sum((d - avg_delta)**2 for d in sample_deltas) / len(sample_deltas)
                        std_dev = variance**0.5
                        if std_dev / avg_delta > 0.05: # Allow 5% variation
                             print(f"Warning: Sample rate appears inconsistent (avg delta: {avg_delta:.6f}, std dev: {std_dev:.6f}). Using average rate.", file=sys.stderr)

                        samples_per_second = round(1.0 / avg_delta) # Round to nearest integer Hz
                        print(f"    Determined Sample Rate: {samples_per_second} Hz (average delta: {avg_delta*1000:.5f} ms)")
                        # --- Stop checking rate after this ---
                        sample_deltas = [] # Clear deltas


                    # Add data to minimap buffer (relative timestamp from start)
                    # Use relative time for minimap, as absolute doesn't matter there
                    relative_timestamp_sec = current_timestamp_sec - first_timestamp_sec
                    minimap_buffer.add_data(current_value_micro_amps, relative_timestamp_sec)

                    # Pack data for session.raw (float 32-bit little-endian current in microamps, unsigned 16-bit little-endian bits for digital data)
                    packed_current = struct.pack('<f', current_value_micro_amps)
                    packed_bits = struct.pack('<H', DEFAULT_DIGITAL_BITS)
                    session_data.write(packed_current)
                    session_data.write(packed_bits)

                    last_timestamp_sec = current_timestamp_sec

                    if row_count % 50000 == 0: # Print progress
                        print(f"    Processed {row_count} rows...")


                except (ValueError, IndexError) as e:
                    # Index error should be caught earlier, but keep for safety
                    print(f"Warning: Skipping row {row_count} due to parsing error: {e}. Row data: {row}", file=sys.stderr)
                    continue

            if samples_per_second is None:
                 if len(sample_deltas) > 0 :
                      # Handle cases with very few samples (< 10)
                      avg_delta = sum(sample_deltas) / len(sample_deltas)
                      if avg_delta <= 1e-9:
                          print(f"Error: Cannot determine sample rate. Average time delta is too small or non-positive ({avg_delta:.3e} s). Check timestamp column.", file=sys.stderr)
                          sys.exit(1)
                      samples_per_second = round(1.0 / avg_delta)
                      print(f"    Determined Sample Rate (few samples): {samples_per_second} Hz (average delta: {avg_delta*1000:.5f} ms)")
                 elif row_count <= 2: # Header + 0 or 1 data row
                      print(f"Error: Cannot determine sample rate. Need at least 2 data rows in CSV.", file=sys.stderr)
                      sys.exit(1)
                 else:
                      # Should not happen if delta check works
                      print(f"Error: Could not determine sample rate after processing all rows.", file=sys.stderr)
                      sys.exit(1)

            total_samples = session_data.tell() // FRAME_SIZE_BYTES
            print(f"  Finished processing {row_count} CSV rows. Total samples: {total_samples}")
            if total_samples == 0:
                print(f"Error: No valid data rows found or processed.", file=sys.stderr)
                sys.exit(1)


    except FileNotFoundError:
        print(f"Error: Input CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # More detailed error
        sys.exit(1)

    # 3. Create PPK2 (ZIP Archive)
    try:
        print(f"  Creating PPK2 file: {output_path}")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            # Create metadata.json
            metadata_content = {
                "metadata": {
                    "samplesPerSecond": samples_per_second,
                    "startSystemTime": start_time_ms_epoch,
                    "createdBy": "csv2ppk2-converter.py",
                    # Add csv specific info
                    "sourceFileInfo": {
                         "filename": os.path.basename(csv_path),
                         "type": "CSV",
                    }
                },
                "formatVersion": PPK2_FORMAT_VERSION
            }
            zf.writestr('metadata.json', json.dumps(metadata_content, indent=None)) # Compact JSON

            # Write session.raw
            session_data.seek(0) # Rewind buffer
            zf.writestr('session.raw', session_data.read())

            # Create minimap.raw
            minimap_content = minimap_buffer.get_state()
            zf.writestr('minimap.raw', json.dumps(minimap_content, indent=None)) # No indent

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
    print(f"  Total time: {duration:.2f} seconds")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert CSV data (with specified timestamp and current columns) to Nordic PPK2 format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
        )

    parser.add_argument('--csv-input', required=True, help='Path to the input CSV file.')
    parser.add_argument('--start-time',
                        help="System start time of the recording in ISO 8601 format (e.g., '2023-10-27T15:30:00Z'). Assumed local time if no timezone provided. If no time provided, current system time is taken.")
    parser.add_argument('--timestamp-column', required=True,
                        help='Name or 0-based index of the timestamp column (in seconds) in the CSV.')
    parser.add_argument('--current-column', required=True,
                        help='Name or 0-based index of the current column (in ampere) in the CSV.')
    parser.add_argument('--output', required=True, help='Path for the output PPK2 file (e.g., my_capture.ppk2).')

    args = parser.parse_args()

    if not args.start_time:
        args.start_time = datetime.now().isoformat()

    # Basic validation before calling main function
    if not args.output.lower().endswith('.ppk2'):
         print("Warning: Output filename does not end with .ppk2", file=sys.stderr)


    convert_csv_to_ppk2(
        args.csv_input,
        args.start_time,
        args.timestamp_column,
        args.current_column,
        args.output
    )
