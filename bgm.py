import cv2
import numpy as np
import imutils
from imutils.video import FPS
import copy
from static_object import *
from intensity_processing import *


####------------------------------------------------------------------------------------------------------------------
def check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals, old_frame, current_frame):
    bbox_to_add = []
    if len(bbox_last_frame_proposals) > 0:
        # print "ciclo vecchie proposte che sono:", len(bbox_last_frame_proposals)
        for old in bbox_last_frame_proposals:
            old_drawn = False
            for curr in bbox_current_frame_proposals:

                if rect_similarity2(old, curr):
                    old_drawn = True
                    break

            if not old_drawn:
                # Check if the area defined by the bounding box in the old frame and in the new one
                # is still the same
                old_section = old_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()
                new_section = current_frame[old[1]:old[1] + old[3], old[0]:old[0] + old[2]].flatten()

                if norm_correlate(old_section, new_section)[0] > 0.95:
                    cv2.rectangle(final_result_image, (old[0], old[1]), (old[0] + old[2], old[1] + old[3]), (255, 0, 0),
                                  1)
                    bbox_to_add.append(old)

    return bbox_to_add


####----------------------------------------------------------------------------------------------------------------


# path1 = r"..\..\dataset\pets\video1.avi"
# cap1 = cv2.VideoCapture(path1)

path = r"videos/aboda/video1.avi"
cap = cv2.VideoCapture(path)
fps = FPS().start()
first_run = True
if (cap.isOpened() == False):
    print("error buka file")
nframe = 0
ret, frame = cap.read()
(height, width, channel) = frame.shape
image_shape = (height, width)
# print(image_shape)
frame1 = imutils.resize(frame, width=450)
(height, width, channel) = frame1.shape
image_shape = (height, width)
# print(image_shape)
# cv2.imshow('frame pertama',frame1)
rgb = IntensityProcessing(image_shape)

bbox_last_frame_proposals = []
static_objects = []

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        frame = imutils.resize(frame, width=450)
        # tampil video FRame
        asli = frame
        cv2.imshow("main", asli)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
        rgb.current_frame = frame
        if first_run:
            old_rgb_frame = copy.copy(rgb.current_frame)
            first_run = False

        rgb.compute_foreground_masks(rgb.current_frame)  # Hitung foreground mask
        rgb.update_detection_aggregator()  # Deteksi jika ada objek yang baru

        rgb_proposal_bbox = rgb.extract_proposal_bbox()  # bounding boxs dari area yang baru
        foreground_rgb_proposal = rgb.proposal_foreground  # rgb proposal

        bbox_current_frame_proposals = rgb_proposal_bbox
        final_result_image = rgb.current_frame.copy()
        old_bbox_still_present = check_bbox_not_moved(bbox_last_frame_proposals, bbox_current_frame_proposals,
                                                      old_rgb_frame, rgb.current_frame.copy())

        # tambahkan bbox yang lama dalam frame yang dideteksi
        bbox_last_frame_proposals = bbox_current_frame_proposals + old_bbox_still_present
        old_rgb_frame = rgb.current_frame.copy()

        draw_bounding_box(final_result_image, bbox_current_frame_proposals)
        draw_bounding_box(foreground_rgb_proposal, rgb_proposal_bbox)

        img = rgb.current_frame
        mask_lg = rgb.foreground_mask_long_term
        mask_sh = rgb.foreground_mask_short_term

        long = cv2.bitwise_and(img, rgb.current_frame, mask=mask_lg)
        cv2.imshow("long", long)

        short = cv2.bitwise_and(img, rgb.current_frame, mask=mask_sh)
        cv2.imshow("short", short)
        # final_result_image = cv2.cvtColor(final_result_image, cv2.COLOR_GRAY2RGB)
        cv2.imshow('Final result', final_result_image)

        cv2.imshow('FG rgb proposal', foreground_rgb_proposal)
        fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break
cap.release()
cv2.destroyAllWindows()

jumlah = len(rgb_proposal_bbox)
print("jumlah statik objek :", jumlah)
for x in static_objects:
    x.print_object()
