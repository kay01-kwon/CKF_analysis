#!/usr/bin/env python3
"""
bag_to_csv.py — ROS2 bag to CSV converter

Reads bag files from Bag/{RPM}/test{N}/ and saves each topic as a CSV file.
Output structure:
    csv_output/
    ├── 4000RPM/
    │   ├── test1/
    │   │   ├── actual_rpm.csv
    │   │   ├── cmd_raw.csv
    │   │   ├── rotor_state.csv
    │   │   └── rotor_state_cov.csv
    │   ├── test2/
    │   │   └── ...
    │   └── ...
    └── 5000RPM/
        └── ...

Usage:
    python bag_to_csv.py Bag
    python bag_to_csv.py Bag --rpm-folders 4000RPM 5000RPM --num-tests 5
    python bag_to_csv.py Bag --output-dir ./csv_output
"""

import os
import csv
import argparse
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def read_bag_to_dict(bag_path):
    """Read a ROS2 bag and return raw data as dict of lists."""
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path, storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    print(f'  Reading: {bag_path}')
    print(f'  Topics: {", ".join(type_map.keys())}')

    data = {
        'actual_rpm': [],
        'cmd_raw': [],
        'rotor_state': [],
        'rotor_state_cov': [],
    }

    while reader.has_next():
        topic, raw_data, time_stamp = reader.read_next()

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw_data, msg_type)
        t = time_stamp * 1e-9

        if topic == '/uav/actual_rpm':
            data['actual_rpm'].append([t, msg.rpm])
        elif topic == '/uav/cmd_raw':
            data['cmd_raw'].append([t, msg.cmd_raw])
        elif topic == '/uav/single_rotor_state':
            data['rotor_state'].append([t, msg.rpm, msg.acceleration])
        elif topic == '/uav/single_rotor_state_covariance':
            data['rotor_state_cov'].append([t, msg.diag_cov[0], msg.diag_cov[1]])

    return data


def save_csv(filepath, header, rows):
    """Save rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def export_bag_to_csv(bag_path, output_dir):
    """Read one bag and export all topics to CSV files."""
    data = read_bag_to_dict(bag_path)

    counts = {}

    if data['actual_rpm']:
        path = os.path.join(output_dir, 'actual_rpm.csv')
        save_csv(path, ['time_s', 'rpm'], data['actual_rpm'])
        counts['actual_rpm'] = len(data['actual_rpm'])

    if data['cmd_raw']:
        path = os.path.join(output_dir, 'cmd_raw.csv')
        save_csv(path, ['time_s', 'cmd_raw'], data['cmd_raw'])
        counts['cmd_raw'] = len(data['cmd_raw'])

    if data['rotor_state']:
        path = os.path.join(output_dir, 'rotor_state.csv')
        save_csv(path, ['time_s', 'rpm', 'acceleration'], data['rotor_state'])
        counts['rotor_state'] = len(data['rotor_state'])

    if data['rotor_state_cov']:
        path = os.path.join(output_dir, 'rotor_state_cov.csv')
        save_csv(path, ['time_s', 'cov_rpm', 'cov_acc'], data['rotor_state_cov'])
        counts['rotor_state_cov'] = len(data['rotor_state_cov'])

    return counts


def main():
    parser = argparse.ArgumentParser(description='ROS2 Bag to CSV converter')

    parser.add_argument('base_dir', type=str,
                        help='Base directory containing RPM folders (e.g., Bag)')
    parser.add_argument('--rpm-folders', nargs='+', type=str,
                        default=['4000RPM', '5000RPM'],
                        help='RPM folder names (default: 4000RPM 5000RPM)')
    parser.add_argument('--num-tests', type=int, default=5,
                        help='Number of test bags per RPM folder (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./csv_output',
                        help='Output directory for CSV files (default: ./csv_output)')

    args = parser.parse_args()

    print(f'=== Bag to CSV Converter ===')
    print(f'Input:  {args.base_dir}')
    print(f'Output: {args.output_dir}\n')

    total_bags = 0

    for rpm in args.rpm_folders:
        for i in range(1, args.num_tests + 1):
            bag_path = os.path.join(args.base_dir, rpm, f'test{i}')
            if not os.path.exists(bag_path):
                print(f'  [SKIP] {bag_path} not found')
                continue

            csv_dir = os.path.join(args.output_dir, rpm, f'test{i}')
            counts = export_bag_to_csv(bag_path, csv_dir)

            count_str = ', '.join(f'{k}: {v}' for k, v in counts.items())
            print(f'  -> Saved to {csv_dir}/ ({count_str})\n')
            total_bags += 1

    print(f'Done! {total_bags} bags exported to {args.output_dir}/')


if __name__ == '__main__':
    main()