import numpy as np
from dataclasses import dataclass, field
from typing import List
from bytetrack.byte_tracker import STrack


@dataclass
class Track:
    class_id: int
    track_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TrackResults:
    tracks: List[Track] = field(default_factory=list)

    @classmethod
    def from_list(cls, track_list: List[STrack]) -> "TrackResults":
        tracks = []
        for track in track_list:
            class_id = int(track.class_id)
            track_id = int(track.track_id)
            score = float(track.score)
            x1 = float(track.tlbr[0])
            y1 = float(track.tlbr[1])
            x2 = float(track.tlbr[2])
            y2 = float(track.tlbr[3])

            tracks.append(
                Track(
                    class_id=class_id,
                    track_id=track_id,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return cls(tracks=tracks)

    def __iter__(self):
        return iter(self.tracks)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        if not self.tracks:
            raise IndexError("No tracks available. The list is empty.")
        return self.tracks[index]

    def __str__(self):
        if not self.tracks:
            return "There are no tracks generated from the current detection results.\n"

        result = ""
        for i, track in enumerate(self.tracks):
            result += (
                f"class_id={track.class_id}, "
                f"track_id={track.track_id}, "
                f"score={track.score:.2f}, "
                f"bbox=({track.x1:.1f}, {track.y1:.1f}, {track.x2:.1f}, {track.y2:.1f})\n"
            )
        return result
