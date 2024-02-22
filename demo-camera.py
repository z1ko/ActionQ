import torch
import lightning
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pprint
import einops as ein
import cv2
import mediapipe
import time

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

args = parser.parse_args()
pprint.pprint(vars(args))

# Load checkpoint
model = ActionQ.load_from_checkpoint(
    checkpoint_path = args.checkpoint,
    maximum_score=30.0,
    model = LRUModel(
        joint_features=4,
        joint_count=len(UPPER_BODY_JOINTS),
        joint_expansion=128,
        temporal_layers_count=4,
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
state = torch.complex(torch.zeros(128), torch.zeros(128))
state = ein.repeat(state, 'c -> t j c', t=4, j=len(UPPER_BODY_JOINTS))
state = state.to(device)

iteration = 0
scores = []

poser_times = []
model_times = []

#def score_update(score, fig, plot):
#    scores.append(score)
#    plot.set_data(scores)
#    fig.gca().relim()
#    fig.gca().autoscale_view()
#    return plot,

#plt.ion()
#plt.show()

while True:
    ret, frame = video.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # frame skeleton (2 step window)
    skeletons = torch.zeros((2, 33, 2))

    frame.flags.writeable = False
    start = time.perf_counter()
    results = poser.process(frame)
    poser_times.append(time.perf_counter() - start)
    frame.flags.writeable = True

    if results.pose_landmarks:
                
        # Draw and save joints
        for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
            cx, cy = landmark.x, landmark.y

            skeletons[0] = skeletons[1]
            skeletons[1, landmark_id] = torch.tensor([cx, cy])
            
            cv2.circle(
                frame, 
                (int(landmark.x * base_width), int(landmark.y * base_height)), 
                5, 
                (255,0,0), 
                cv2.FILLED
            )

        mediapipe.solutions.drawing_utils.draw_landmarks(
            frame,
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
            scores.append(score)

            #plt.plot(scores, color='b')
            #plt.draw()
            #plt.pause(0.001)

    cv2.imshow('video', frame)
    if cv2.waitKey(1000//int(fps)) & 0xFF == ord('q'):
        break

    iteration += 1

#plt.ioff()

plt.plot(scores)
plt.title('model scores')
plt.ylim(0.0, 30.0)
plt.show()

plt.plot(poser_times)
plt.title('mediapipe poser time')
plt.show()

plt.plot(model_times)
plt.title('model time')
plt.show()