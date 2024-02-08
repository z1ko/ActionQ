import mediapipe
import cv2
import argparse
import pprint
import os
import time

def process_video(filepath, poser, args):
    video = cv2.VideoCapture(filepath)
    pTime = 0
    while video.isOpened():

        ret, frame = video.read()
        if not ret:
            break

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = poser.process(frame)
        if results.pose_landmarks:
            mediapipe.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mediapipe.solutions.pose.POSE_CONNECTIONS
            )

            #for id, lm in enumerate(results.pose_landmarks.landmark):
            #    h, w, c = frame.shape
            #    print(id, lm)
            #
            #    cx, cy = int(lm.x*w), int(lm.y*h)
            #    cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            
            #cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
    

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='data/raw/capture')
parser.add_argument('--output_folder', type=str, default='data/raw/skeleton')

args = parser.parse_args()
pprint.pprint(vars(args))

poser = mediapipe.solutions.pose.Pose()
for file in os.scandir(args.input_folder):
    if file.name.endswith('.avi'):
        process_video(file.path, poser, args)