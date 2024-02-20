
import argparse
import pprint
import cv2
import time
import os
import mediapipe
import numpy as np
import pickle
import random
import string

VERSION = '0.1.5'

# For the homemade training dataset we use only the upper body joints
#UPPER_BODY_JOINT = frozenset([11,12,13,14,15,16])
#UPPER_BODY_CONNECTIONS = frozenset([(12,11),(11,13),(12,14),(13,15),(14,16)])

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--lenght', type=int, default=60)
parser.add_argument('--save_dir', type=str, default='data/raw/capture')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--repetition_delay', type=int, default=5)
parser.add_argument('--draw_skeleton', default=False, action='store_true')

args = parser.parse_args()
pprint.pprint(vars(args))

video = cv2.VideoCapture(args.camera, cv2.CAP_V4L)
base_width, base_height, fps = (
    video.get(cv2.CAP_PROP_FRAME_WIDTH),
    video.get(cv2.CAP_PROP_FRAME_HEIGHT),
    video.get(cv2.CAP_PROP_FPS),
)
print(f'camera(width: {base_width}, height: {base_height}, fps: {fps})')

print(f'forcing resolution to {args.width}x{args.height}')
video.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

# Unique id for this session
id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
folder = os.path.join(args.save_dir, VERSION, id)
os.makedirs(folder)

poser = mediapipe.solutions.pose.Pose()

exit_requested = False
for rep in range(args.repetitions):
    if exit_requested:
        break

    print(f'REPETITION N. {rep}, START!')

    frames = []
    skeleton_frames = []

    frame_idx = 0
    start_time = time.time_ns()
    while (time.time_ns() - start_time) * 1e-9 <= args.lenght: # secondi
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit_requested = True
            break

        #print(f'repetition-{rep}-frame-{frame_idx}')
        frame = cv2.flip(frame, 1)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame.copy())

        # We save only the upper body joints and their 2D position on the screen
        skeleton = np.zeros((33, 2))

        frame.flags.writeable = False
        results = poser.process(frame)
        frame.flags.writeable = True

        if results.pose_landmarks:
                
            # Draw and save joints
            skeleton_id = 0
            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = landmark.x, landmark.y
                skeleton[skeleton_id] = np.array([cx, cy])
                skeleton_id += 1 # Save all skeleton joints
                    
                if args.draw_skeleton:
                    cv2.circle(
                        frame, 
                        (int(landmark.x * args.width), int(landmark.y * args.height)), 
                        5, 
                        (255,0,0), 
                        cv2.FILLED
                    )

            # Draw skeleton
            if args.draw_skeleton:
                mediapipe.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mediapipe.solutions.pose.POSE_CONNECTIONS
                )

            skeleton_frames.append(skeleton)

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('video', frame)
        frame_idx += 1

        if cv2.waitKey(1000//int(fps)) & 0xFF == ord('q'):
            exit_requested = True 
            break
   
    # Ask user for quality of repetition
    # NOTE: hardcoded
    #cf1 = input('è stato raggiunto l\'apice del movimento [0-5]: ')
    #cf2 = input('è stata raggiunta la posizione di riposo ad ogni ripetizione [0-5]: ')
    #cf3 = input('la postura è stata mantenuta correttamente [0-5]: ')
    #cf4 = input('la velocità dell\'esercizio ha seguito un andamento normale [0-5]: ')
    #cf5 = input('è stato eseguito in modo simmetrico [0-5]: ')
    #cf6 = input('la ripetizione è state eseguita correttamente con le braccia [0-5]: ')

    cf = input('il movimento è stato eseguito correttamente? [0: no, 1: più no che si, 2: più si che no, 3: si]: ')

    framerate = len(frames) / args.lenght # framerate effettivo
    series_folder = os.path.join(folder, f'rep-{rep:03}-frames-{len(frames)}-fps-{framerate}')
    os.makedirs(series_folder)

    print(f'saving repetition in {series_folder}')

    # write control factors
    filepath = os.path.join(series_folder, 'control_factors.csv')
    with open(filepath, 'wt') as f:
        f.write(cf) # cf1, cf2, cf3, cf4, cf5

    # write all frames
    filepath = os.path.join(series_folder, 'video.avi')
    writer = cv2.VideoWriter(filepath, fourcc, framerate, (args.width, args.height))
    for frame in frames: writer.write(frame)
    writer.release()

    # Write all skeleton data
    filepath = os.path.join(series_folder, 'skeleton.pkl')
    skeleton_merged = np.stack(skeleton_frames) # (frames, joints, features)
    with open(filepath, 'wb') as f:
        pickle.dump(skeleton_merged, f)

    if rep != args.repetitions - 1:
        print('READY FOR NEXT REPETITION...')
        time.sleep(float(args.repetition_delay))

video.release()
cv2.destroyAllWindows()
