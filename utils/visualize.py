#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np

from utils.classes import COCO_CLASSES
from dataclass.track_results import TrackResults


def vis(img, *args, conf=0.3, class_names=COCO_CLASSES):
    if isinstance(args[0], TrackResults):
        class_ids, track_ids, scores, boxes = [[] for _ in range(4)]
        for track in args[0].tracks:
            class_ids.append(track.class_id)
            track_ids.append(track.track_id)
            scores.append(track.score)
            boxes.append([track.x1, track.y1, track.x2, track.y2])
    else:
        class_ids, track_ids, scores, boxes = args

    for i in range(len(boxes)):
        class_id = int(class_ids[i])
        if track_ids:
            track_id = int(track_ids[i])
        score = scores[i]
        box = boxes[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_color = (0, 0, 0) if np.mean(_COLORS[class_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if track_ids:
            det_text = f"{class_names[class_id]}: {score * 100:.1f}%"
            track_text = f"tracklet {track_id}"

            det_txt_size = cv2.getTextSize(det_text, font, 0.4, 1)[0]
            track_txt_size = cv2.getTextSize(track_text, font, 0.4, 1)[0]

            det_txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0),
                (x0 + det_txt_size[0] + 5, y0 + det_txt_size[1] + 5),
                det_txt_bk_color,
                -1,
            )
            cv2.putText(
                img,
                det_text,
                (x0 + 2, y0 + det_txt_size[1]),
                font,
                0.4,
                txt_color,
                thickness=1,
            )

            track_txt_bk_color = (
                (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
            )
            cv2.rectangle(
                img,
                (x0, y1 - track_txt_size[1] - 5),
                (x0 + track_txt_size[0] + 5, y1),
                track_txt_bk_color,
                -1,
            )
            cv2.putText(
                img, track_text, (x0 + 2, y1 - 5), font, 0.4, txt_color, thickness=1
            )

        else:
            text = f"{class_names[class_id]}: {score * 100:.1f}%"

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1,
            )
            cv2.putText(
                img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1
            )

    return img


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
