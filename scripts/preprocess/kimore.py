
import copy
import pandas as pd
import os

# All available subjects in the dataset
SUBJECT_TYPES = {
    'expert': 'CG/Expert',
    'non-expert': 'CG/NotExpert',
    'stroke': 'GPP/Stroke',
    'parkinson': 'GPP/Parkinson',
    'backpain': 'GPP/BackPain'
}

# Where the dataset is stored
ROOT_DIR = "data/KiMoRe"

# Windows size for frames
WINDOW_SIZE = 100

# Joints of the body
JOINTS_COUNT = 25
# Joints that are not usefull (hands, facial, feet)
EXCLUDED_JOINTS = [15, 20, 21, 22, 23, 24]


def _load_single_exercise(samples, data_descriptor, filepath):
    with open(filepath) as f:
        frame = 0
        for line in f.readlines():
            if len(line) >= 10:
                data_descriptor['frame'] = frame
                frame += 1

                tokens = line.split(',')[:-1]
                for joint in range(JOINTS_COUNT):
                    if joint in EXCLUDED_JOINTS:
                        continue

                    data_descriptor['joint'] = joint

                    # Insert data unit
                    data_descriptor['pos_x'] = float(tokens[joint * 4 + 0]),
                    data_descriptor['pos_y'] = float(tokens[joint * 4 + 1]),
                    data_descriptor['pos_z'] = float(tokens[joint * 4 + 2]),

                    # Merge data
                    for key, value in samples.items():
                        samples[key].append(data_descriptor[key])


def _load_evaluations(targets, data_descriptor, filepath):
    with open(filepath) as f:
        _, values = f.readline(), f.readline()
        tokens = values.split(',')
        print(tokens)
        for exercise in range(5):  # TODO: Maybe data is broken, check correct number of elements

            eval_descriptor = copy.deepcopy(data_descriptor)
            eval_descriptor['exercise'] = exercise
            eval_descriptor['TS'] = float(tokens[1 + 0 + exercise])
            eval_descriptor['PO'] = float(tokens[1 + 5 + exercise])
            eval_descriptor['CF'] = float(tokens[1 + 10 + exercise])

            # Merge data
            for key, value in targets.items():
                targets[key].append(eval_descriptor[key])


# Resulting samples
samples = {

    # Metadata
    'type': [],
    'subject': [],
    'exercise': [],
    'frame': [],
    'joint': [],

    # Effective data
    'pos_x': [],
    'pos_y': [],
    'pos_z': []
}

# Resulting target for each exercise
targets = {

    # Metadata
    'type': [],
    'subject': [],
    'exercise': [],

    # Effective data
    'TS': [],
    'PO': [],
    'CF': []
}

# Dict for a single data entry
data_descriptor = {}

# Load all data
for subject_type in SUBJECT_TYPES.values():
    data_descriptor['type'] = subject_type

    path = os.path.join(ROOT_DIR, subject_type)
    subjects_list = [f.name for f in os.scandir(path) if f.is_dir()]
    for subject in subjects_list:
        data_descriptor['subject'] = subject

        # The dataset is strange, all exercises folders have the same label file containing the evaluations
        # of all the exercises, such a waste of space. Process only the first one
        loaded_evaluations = False

        # Process all exercises
        subject_path = os.path.join(path, subject)
        exercises_list = [f.name for f in os.scandir(subject_path) if f.is_dir()]
        for exercise in exercises_list:
            data_descriptor['exercise'] = int(exercise[-1:])

            print(f'processing {subject_type}/{subject}/{exercise}')
            exercise_path = os.path.join(subject_path, exercise)

            # Process evaluation data
            if not loaded_evaluations:
                exercises_eval_path = os.path.join(exercise_path, 'Label')
                if os.path.exists(exercises_eval_path):
                    for file in os.scandir(exercises_eval_path):
                        if file.name.startswith('ClinicalAssessment') and file.name.endswith('.csv'):
                            _load_evaluations(targets, data_descriptor, file.path)
                            loaded_evaluations = True
                            break

            # Process frame data
            exercise_raw_path = os.path.join(exercise_path, 'Raw')
            if os.path.exists(exercise_raw_path):
                for file in os.scandir(exercise_raw_path):
                    if file.name.startswith('JointPosition'):
                        _load_single_exercise(samples, data_descriptor, file.path)


# Convert dict to dataframe for faster loading and analysis
data_df = pd.DataFrame.from_dict(samples)
data_df.to_parquet('data/processed/kimore_samples.parquet.gzip', compression='gzip')
print(f'processed KiMoRe samples:\n{data_df}')

targets_df = pd.DataFrame.from_dict(targets)
targets_df.to_parquet('data/processed/kimore_targets.parquet.gzip', compression='gzip')
print(f'processed KiMoRe targets:\n{targets_df}')
