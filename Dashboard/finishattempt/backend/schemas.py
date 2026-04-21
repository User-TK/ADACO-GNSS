from pydantic import BaseModel
from typing import List

class GNSSInput(BaseModel):
    # scalar features — same 30 as your model expects
    nav_pvt:      List[float]  # 15 values
    nav_clock:    List[float]  # 4 values
    nav_dop:      List[float]  # 7 values
    nav_posecef:  List[float]  # 4 values
    # spectrum — 2 RF blocks x 256 bins
    spectrum_01:  List[float]  # 256 values
    spectrum_02:  List[float]  # 256 values

class PredictionOutput(BaseModel):
    label:       int            # 0=Clean, 1=Spoofing, 2=Jamming
    label_name:  str
    confidence:  float          # probability of predicted class
    probabilities: List[float]  # [P(clean), P(spoofing), P(jamming)]