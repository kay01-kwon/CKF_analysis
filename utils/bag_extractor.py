#!/usr/bin/env python3

import os
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


class BagExtractor:
    """
    Extracts bag data from a rosbag file
    """

    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.data = {
            'cmd_raw': {'t': [], 'raw': []},
            'actual_rpm': {'t': [], 'rpm': []},
            'rotor_state': {'t': [], 'rpm': [], 'acceleration': []},
            'rotor_state_cov': {'t': [], 'cov': []}
        }

    def read_bag(self):
        """Read ros2 bag and extract data from specified topics"""
        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_file,
            storage_id='sqlite3'
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        print(f'Reading bag: {self.bag_file}')
        print('Available topics:')
        for topic in topic_types:
            print(f'  {topic.name}: {topic.type}')

        while reader.has_next():
            topic, data, time_stamp = reader.read_next()

            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            # Convert nanoseconds to seconds
            t = time_stamp * 1e-9

            if topic == '/uav/actual_rpm':
                self.data['actual_rpm']['t'].append(t)
                self.data['actual_rpm']['rpm'].append(msg.rpm)
            elif topic == '/uav/cmd_raw':
                self.data['cmd_raw']['t'].append(t)
                self.data['cmd_raw']['raw'].append(msg.cmd_raw)
            elif topic == '/uav/single_rotor_state':
                self.data['rotor_state']['t'].append(t)
                self.data['rotor_state']['rpm'].append(msg.rpm)
                self.data['rotor_state']['acceleration'].append(msg.acceleration)
            elif topic == '/uav/single_rotor_state_covariance':
                self.data['rotor_state_cov']['t'].append(t)
                self.data['rotor_state_cov']['cov'].append(
                    [msg.diag_cov[0], msg.diag_cov[1]]
                )

        # Convert lists to numpy arrays AFTER the loop, using self.data
        for key in self.data:
            for subkey in self.data[key]:
                self.data[key][subkey] = np.array(self.data[key][subkey])

        return self.data


def extract_all_bags(base_dir, rpm_folders=None, num_tests=5):
    """
    Extract bag data from multiple RPM folders.

    Args:
        base_dir: Base directory (e.g., 'CKF')
        rpm_folders: List of RPM folder names (default: ['4000RPM', '5000RPM'])
        num_tests: Number of test bags per RPM folder

    Returns:
        dict: {rpm_folder: [data_test1, data_test2, ...]}
    """
    if rpm_folders is None:
        rpm_folders = ['4000RPM', '5000RPM']

    all_data = {}

    for rpm in rpm_folders:
        all_data[rpm] = []
        for i in range(1, num_tests + 1):
            bag_path = os.path.join(base_dir, rpm, f'test{i}')
            if not os.path.exists(bag_path):
                print(f'[WARNING] Bag not found: {bag_path}, skipping.')
                continue

            extractor = BagExtractor(bag_path)
            data = extractor.read_bag()
            all_data[rpm].append(data)
            print(f'  -> Extracted {rpm}/test{i}: '
                  f'{len(data["actual_rpm"]["t"])} actual_rpm msgs, '
                  f'{len(data["rotor_state"]["t"])} rotor_state msgs\n')

    return all_data


if __name__ == '__main__':
    # Example usage
    all_data = extract_all_bags('CKF', rpm_folders=['4000RPM', '5000RPM'], num_tests=5)

    for rpm, tests in all_data.items():
        print(f'\n=== {rpm}: {len(tests)} bags loaded ===')
        for i, data in enumerate(tests):
            print(f'  test{i+1}: {len(data["rotor_state"]["t"])} rotor_state samples')