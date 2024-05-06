import torch
import lightning
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
import numpy as np
import argparse
import pprint
import einops as ein
import cv2
import mediapipe
import time
import os

from actionq.dataset.camera import SkeletonSubset, JointRelativePosition, JointDifference
from actionq.utils.transform import Compose
from actionq.model.lru_model import LRUModel
from actionq.model.regression import ActionQ

UPPER_BODY_JOINTS=[11,12,13,14,15,16]
UPPER_BODY_JOINTS_ROOTS=[11,12]

# TODO: Save network data (1, 1, F)
STANDARDIZE_MEAN=torch.tensor([[[3.6398e-04, 3.4543e-02, 1.1676e-06, 3.4535e-05]]])
STANDARDIZE_STD=torch.tensor([[[0.1193, 0.1159, 0.0139, 0.0277]]])

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)
parser.add_argument('lenght', type=int, default=10)

args = parser.parse_args()
pprint.pprint(vars(args))

# Load checkpoint
model = ActionQ.load_from_checkpoint(
    checkpoint_path = args.checkpoint,
    maximum_score=35.0,
    model = LRUModel(
        joint_features=4,
        joint_count=len(UPPER_BODY_JOINTS),
        joint_expansion=256,
        temporal_layers_count=2,
        spatial_layers_count=0,
        output_dim=1,
        dropout=0.2,
        r_min=0.4,
        r_max=0.8,
        mode='predict'
    )
)

model.eval().to(torch.device('cuda'))
device = torch.device('cuda')

video = cv2.VideoCapture(0, cv2.CAP_V4L)
base_width, base_height, fps = (
    video.get(cv2.CAP_PROP_FRAME_WIDTH),
    video.get(cv2.CAP_PROP_FRAME_HEIGHT),
    video.get(cv2.CAP_PROP_FPS),
)

poser = mediapipe.solutions.pose.Pose()

skeleton_root_joints = [UPPER_BODY_JOINTS.index(j) for j in UPPER_BODY_JOINTS_ROOTS]
transform = Compose([
    SkeletonSubset(UPPER_BODY_JOINTS),
    JointRelativePosition(skeleton_root_joints),
    JointDifference()
])

# State of the system
state = torch.complex(torch.zeros(256), torch.zeros(256))
state = ein.repeat(state, 'c -> t j c', t=2, j=len(UPPER_BODY_JOINTS))
state = state.to(device)

iteration = 0

times = []
scores = []

poser_times = []
model_times = []

def update_output_data():
    dpg.configure_item('line', x=times, y=scores)
    dpg.fit_axis_data("xaxis")

def update_camera(frame):
    global gui_frame
    gui_frame[:, :, :] = frame.astype(np.float32) / 255.0

def update_skeleton(skeleton):
    global gui_skeleton
    gui_skeleton[:, :, :] = skeleton.astype(np.float32) / 255.0

# Load layout configuration if present
init_file = None
if os.path.exists("custom_layout.ini"):
    init_file = 'custom_layout.ini'

dpg.create_context()
dpg.configure_app(docking=True, docking_space=True, init_file=init_file)

# Create dynamic texture from video and skeleton output
gui_frame = np.zeros((int(base_height), int(base_width), 3), dtype=np.float32)
gui_skeleton = np.zeros((int(base_height), int(base_width), 3), dtype=np.float32)
with dpg.texture_registry(show=False):

    dpg.add_raw_texture(
        width=int(base_width), 
        height=int(base_height), 
        default_value=gui_frame, 
        format=dpg.mvFormat_Float_rgb, 
        tag="camera")

    dpg.add_raw_texture(
        width=int(base_width), 
        height=int(base_height), 
        default_value=gui_skeleton, 
        format=dpg.mvFormat_Float_rgb, 
        tag="skeleton")

# Create output window
with dpg.window(width=600, height=300, label="Model's scores", tag='model_output'):
    with dpg.plot(width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="xaxis", time=True, no_tick_labels=True)
        dpg.add_plot_axis(dpg.mvYAxis, label="Score", tag="yaxis")
        dpg.add_line_series([], [], tag='line', parent="yaxis")
        dpg.set_axis_limits("yaxis", 0.0, 35.0)

with dpg.window(label='Configuration'):
    dpg.add_button(label="Save layout", callback=lambda: dpg.save_init_file("custom_layout.ini"))

# Create video window
with dpg.window(label="Camera", tag='camera_output'):
    dpg.add_image('camera')
    
with dpg.window(label="Skeleton", tag='skeleton_ouput'):
    dpg.add_image('skeleton')

dpg.create_viewport(title='ActionQ')
dpg.setup_dearpygui()
dpg.show_viewport(maximized=True)
while dpg.is_dearpygui_running():

    start_time = time.time_ns()
    while (time.time_ns() - start_time) * 1e-9 <= args.lenght:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)

        # frame skeleton (2 step window)
        skeletons = torch.zeros((2, 33, 2))

        frame.flags.writeable = False
        start = time.perf_counter()
        results = poser.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        poser_times.append(time.perf_counter() - start)
        frame.flags.writeable = True

        skeleton_image_tmp = np.zeros_like(gui_skeleton)
        if results.pose_landmarks:
                    
            # Draw and save joints
            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = landmark.x, landmark.y

                skeletons[0] = skeletons[1]
                skeletons[1, landmark_id] = torch.tensor([cx, cy])
                
                cv2.circle(
                    skeleton_image_tmp, 
                    (int(landmark.x * base_width), int(landmark.y * base_height)), 
                    5, 
                    (255,0,0), 
                    cv2.FILLED
                )

            mediapipe.solutions.drawing_utils.draw_landmarks(
                skeleton_image_tmp,
                results.pose_landmarks,
                mediapipe.solutions.pose.POSE_CONNECTIONS
            )

            # send skeleton to model in real-time
            if iteration != 0:
                
                # feature augmentation and standardization
                data = transform(skeletons)
                data = (data - STANDARDIZE_MEAN) / STANDARDIZE_STD
                data.squeeze_(0)

                start = time.perf_counter()
                score = model.predict_step(data.to(device), state).item()
                model_times.append(time.perf_counter() - start)

                print(f'score for current frame: {score}')
                times.append(time.time())
                scores.append(score)

                update_camera(frame)
                update_skeleton(skeleton_image_tmp)
                update_output_data()

        #cv2.imshow('video', frame)
        if cv2.waitKey(1000//int(fps)) & 0xFF == ord('q'):
            dpg.stop_dearpygui()
            break

        dpg.render_dearpygui_frame()
        iteration += 1

    # No more time
    dpg.stop_dearpygui()

# Destroy output window
dpg.destroy_context()

plt.plot(scores)
plt.title('model scores')
plt.ylim(0.0, 35.0)
plt.show()

plt.plot(poser_times)
plt.title('mediapipe poser time')
plt.show()

plt.plot(model_times)
plt.title('model time')
plt.show()