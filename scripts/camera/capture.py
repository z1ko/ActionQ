
import argparse
import pprint
import cv2
import time
import os
import mediapipe
import numpy as np
import pickle

# For the homemade training dataset we use only the upper body joints
UPPER_BODY_JOINT = frozenset([11,12,13,14,15,16])
UPPER_BODY_CONNECTIONS = frozenset([(12,11),(11,13),(12,14),(13,15),(14,16)])

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--lenght', type=int, default=60)
parser.add_argument('--save_dir', type=str, default='data/raw/capture')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--repetition_delay', type=int, default=5)
parser.add_argument('--pose_estimator', type=bool, default=True)

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

if args.pose_estimator:
    poser = mediapipe.solutions.pose.Pose()

exit_requested = False
for rep in range(args.repetitions):
    if exit_requested:
        break

    print(f'REPETITION N. {rep}')

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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract skeleton
        if args.pose_estimator:

            # We save only the upper body joints and their 2D position on the screen
            skeleton = np.zeros((len(UPPER_BODY_JOINT), 2))

            frame.flags.writeable = False
            results = poser.process(frame)
            frame.flags.writeable = True

            if results.pose_landmarks:
                
                # Draw and save joints
                skeleton_id = 0
                for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark_id in UPPER_BODY_JOINT:
                        cx, cy = landmark.x, landmark.y
                        skeleton[skeleton_id] = np.array([cx, cy])
                        skeleton_id += 1

                        cv2.circle(
                            frame, 
                            (int(landmark.x * args.width), int(landmark.y * args.height)), 
                            5, 
                            (255,0,0), 
                            cv2.FILLED
                        )

                # Draw skeleton
                mediapipe.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    #mediapipe.solutions.pose.POSE_CONNECTIONS
                    UPPER_BODY_CONNECTIONS
                )

            skeleton_frames.append(skeleton)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('video', frame)
        frames.append(frame)
        frame_idx += 1

        if cv2.waitKey(1000//int(fps)) & 0xFF == ord('q'):
            exit_requested = True 
            break

    # write all frames
    filename = os.path.join(args.save_dir, f'repetition-{rep}')
    framerate = len(frames) / args.lenght # framerate effettivo
    print(f'saving repetition to {filename}, frames={len(frames)}, framerate={framerate}')
    writer = cv2.VideoWriter(f'{filename}.avi', fourcc, framerate, (args.width, args.height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    # Write all skeleton data
    skeleton_merged = np.stack(skeleton_frames) # (frames, joints, features)
    with open(f'{filename}.skeleton.pkl', 'wb') as f:
        pickle.dump(skeleton_merged, f)

    if rep != args.repetitions - 1:
        print('READY FOR NEXT REPETITION')
        time.sleep(float(args.repetition_delay))

video.release()
cv2.destroyAllWindows()