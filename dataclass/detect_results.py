import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResults:
    detections: List[Detection] = field(default_factory=list)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "DetectionResults":
        detections = []
        for row in arr:
            # row: [class_id, score, x1, y1, x2, y2]
            class_id = int(row[0])
            score = float(row[1])
            x1 = float(row[2])
            y1 = float(row[3])
            x2 = float(row[4])
            y2 = float(row[5])

            detections.append(
                Detection(class_id=class_id, score=score, x1=x1, y1=y1, x2=x2, y2=y2)
            )
        return cls(detections=detections)

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index):
        if not self.detections:
            raise IndexError("No detections available. The list is empty.")
        return self.detections[index]

    def __str__(self):
        if not self.detections:
            return "No objects were detected in the current frame.\n"

        result = ""
        for i, det in enumerate(self.detections):
            result += (
                f"Detection {i}: "
                f"class_id={det.class_id}, "
                f"score={det.score:.2f}, "
                f"bbox=({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f})\n"
            )

        return result
