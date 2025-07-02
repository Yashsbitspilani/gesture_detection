# # air_canvas.py
# import argparse
# import cv2
# import numpy as np
# import time

# from hand import HandDetector        # your detector
# from utils.utils import get_finger_state  # same util that your GestureDetector uses

# # -------------------------------------------------------------------
# # Configuration
# # -------------------------------------------------------------------
# CAM_W, CAM_H = 1280, 720
# DRAW_COLOR    = (0, 255, 0)   # bright green
# DRAW_THICK    = 5

# # finger-state thresholds (reuse your existing for “up/down”)
# THUMB_THRESH      = [9, 8]
# NON_THUMB_THRESH  = [8.6, 7.6, 6.6, 6.1]
# BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]

# # index finger is finger 1 in your state array
# # fist = all states “down” (0)
# # draw mode = [0,1,0,0,0]  (thumb down, index up, others down)
# DRAW_STATE = [0,1,0,0,0]

# # -------------------------------------------------------------------
# # A minimal finger‐state checker for draw vs. fist
# # -------------------------------------------------------------------
# def compute_finger_states(lm_array, label, facing):
#     """Return a NumPy‐array of states [thumb, index, middle, ring, pinky]."""
#     from utils.utils import calculate_thumb_angle, calculate_angle, two_landmark_distance, get_finger_state

#     states = [None] * 5
#     d1 = two_landmark_distance(lm_array[0], lm_array[5])

#     for i in range(5):
#         joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
#         if i == 0:
#             # Thumb: compute 3 angles, then make it a NumPy array
#             angle_list = [
#                 calculate_thumb_angle(lm_array[joints[j:j+3]], label, facing)
#                 for j in range(3)
#             ]
#             angles = np.array(angle_list)  # <— convert to array
#             states[i] = get_finger_state(angles, THUMB_THRESH)
#         else:
#             # Other fingers
#             angle_list = [
#                 calculate_angle(lm_array[joints[j:j+3]])
#                 for j in range(3)
#             ]
#             angles = np.array(angle_list)  # <— convert to array
#             d2 = two_landmark_distance(lm_array[joints[1]], lm_array[joints[4]])
#             st = get_finger_state(angles, NON_THUMB_THRESH)
#             if st == 0 and (d2/d1) < BENT_RATIO_THRESH[i-1]:
#                 st = 1
#             states[i] = st

#     return states


# # -------------------------------------------------------------------
# # Main Air-Canvas Application
# # -------------------------------------------------------------------
# def main(max_hands=1):
#     cap    = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

#     detector = HandDetector(
#         static_image_mode=False,
#         max_num_hands=max_hands,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.5
#     )

#     # a transparent canvas we draw on
#     canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

#     prev_x, prev_y = None, None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.flip(frame, 1)
#         hands = detector.detect_hands(frame)

#         # default: no drawing
#         drawing = False
#         clear   = False

#         if hands:
#             # just use first hand
#             hand = hands[0]
#             lm   = hand['landmarks']
#             label   = hand['label']
#             facing  = hand['facing']

#             # compute which fingers are up/down
#             states = compute_finger_states(lm, label, facing)

#             # decide mode
#             if states == DRAW_STATE:
#                 drawing = True
#             elif all(s == 0 for s in states):
#                 clear = True

#             # get index tip coords
#             ix, iy = lm[8]  # landmark 8 is index-finger tip

#             # ... after detect_hands and mode logic ...
#             if drawing:
#                 # instead of ix, iy = lm[8]
#                 ix, iy, _ = lm[8]     # <-- grab only x & y
#                 if prev_x is None:
#                     prev_x, prev_y = ix, iy
#                 cv2.line(canvas, (prev_x, prev_y), (ix, iy), DRAW_COLOR, DRAW_THICK)
#                 prev_x, prev_y = ix, iy
#             else:
#                 prev_x, prev_y = None, None


#             if clear:
#                 canvas[:] = 0  # reset canvas

#         # overlay canvas onto frame (semi-transparent)
#         overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

#         cv2.imshow('Air Canvas', overlay)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # -------------------------------------------------------------------
# # Entry Point
# # -------------------------------------------------------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--max_hands', type=int, default=1,
#                         help='how many hands to detect (default: 1)')
#     args = parser.parse_args()
#     main(**vars(args))

import cv2
import numpy as np
import time
import argparse
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser(description='MediaPipe Air Canvas: Draw with hand gestures')
    parser.add_argument('--max_hands', type=int, default=1, help='Maximum number of hands to detect')
    args = parser.parse_args()

    # Initialize webcam and MediaPipe Hands
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=args.max_hands,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.5)
    canvas = None
    prev_x, prev_y = 0, 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)  # create blank canvas on first frame
        h, w, _ = frame.shape

        # Run hand landmark detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            # Process only the first detected hand for drawing logic
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                lm = hand_landmarks.landmark

                # Get fingertip and PIP coordinates
                index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
                pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
                pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]
                thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
                thumb_mcp = lm[mp_hands.HandLandmark.THUMB_MCP]

                # Check which fingers are up
                index_up  = index_tip.y  < index_pip.y
                middle_up = middle_tip.y < middle_pip.y
                ring_up   = ring_tip.y   < ring_pip.y
                pinky_up  = pinky_tip.y  < pinky_pip.y
                if label == 'Right':
                    thumb_up = thumb_tip.x < thumb_mcp.x
                else:
                    thumb_up = thumb_tip.x > thumb_mcp.x

                # Gesture: only index finger up => draw
                if index_up and not thumb_up and not middle_up and not ring_up and not pinky_up:
                    ix = int(index_tip.x * w)
                    iy = int(index_tip.y * h)
                    if prev_x != 0 and prev_y != 0:
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 255, 255), thickness=2)
                    prev_x, prev_y = ix, iy
                # Gesture: all fingers down (fist) => clear
                elif not index_up and not thumb_up and not middle_up and not ring_up and not pinky_up:
                    canvas = np.zeros_like(frame)
                    prev_x, prev_y = 0, 0
                else:
                    prev_x, prev_y = 0, 0

                break
        else:
            # No hand detected: reset drawing state
            prev_x, prev_y = 0, 0

        # Overlay canvas on the frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, canvas)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Air Canvas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

