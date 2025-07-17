"""Define optimized lhereader"""

import re
import io
from dataclasses import dataclass, field
from xml.etree import ElementTree
from skhep.math import LorentzVector


@dataclass
class Particle:
    pdgid: int
    px: float
    py: float
    pz: float
    energy: float
    color_idx: int = None, # optional
    helicity: int = None, # optional

    def p4(self):
        return LorentzVector(self.px, self.py, self.pz, self.energy)


@dataclass
class Event:
    particles: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    scale: float = -1

    def add_particle(self, particle):
        self.particles.append(particle)


class LHEReader:
    def __init__(
        self,
        file_path: str,
        weight_mode: str = "list",
        weight_regex: str = ".*",
    ):
        """_summary_

        Args:
            file_path (str): Path to input LHE file
            weight_mode (str, optional): Format to return weights as. Can be dict or list.
                If dict, weight IDs are used as keys. Defaults to "list".
            weight_regex (str, optional): Regular expression to select weights to be read.
                Defaults to reading all.
        """
        self.file_path = file_path
        with open(self.file_path, "r") as f:
            content = f.read()
        wrapped_content = f"<LesHouchesEvents>\n{content}\n</LesHouchesEvents>"

        self.iterator = ElementTree.iterparse(
            io.StringIO(wrapped_content), events=("start", "end")
        )
        self.current = None
        self.current_weights = None

        assert weight_mode in ["list", "dict"]
        self.weight_mode = weight_mode
        self.weight_regex = re.compile(weight_regex)

    def unpack_from_iterator(self):
        # Read the lines for this event
        lines = self.current[1].text.strip().split("\n")

        # Create a new event
        event = Event()
        # Read header
        event_header = lines[0].strip()
        next = int(event_header.split()[0].strip())
        

        amp_times_weight = float(event_header.split()[2].strip())
        weight = float(event_header.split()[6].strip())
        
        helicity_conf = []
        color_indeces = []
        for i in range(next):
            color_indeces.append(int(lines[1].strip().split()[i].strip()))
            hel_index = int(lines[4].strip().split()[i].strip())
            hel_index = -1 if hel_index == 0 else hel_index
            helicity_conf.append(hel_index)

        r_LC_to_FC = float(lines[2].strip().split()[0].strip())
        r_LC_to_NLC = float(lines[2].strip().split()[1].strip())
        A_LC, A_NLC, A_FC = float(lines[2].split()[2].strip()), float(lines[2].split()[3].strip()), float(lines[2].split()[4].strip())

        # weight_LC = float(lines[4].strip().split()[0].strip())
        # weight_NLC = float(lines[4].strip().split()[1].strip())
        # weight_FC = float(lines[4].strip().split()[2].strip())

        

        event.amp_times_weight = amp_times_weight
        event.A_LC = A_LC
        event.A_NLC = A_NLC
        event.A_FC = A_FC
        event.weight = weight
        event.r_LC_to_FC = r_LC_to_FC
        event.r_LC_to_NLC = r_LC_to_NLC
        event.weight_LC = None # weight_LC
        event.weight_NLC = None # nullweight_NLC
        event.weight_FC = None # nullweight_FC
        event.helicity_conf = helicity_conf
        event.color_indeces = color_indeces

        # Iterate over particle lines and push back
        for ipart in range(4 + 1, 4 + next + 1):
            part_data = lines[ipart].strip().split()
            p = Particle(
                pdgid=int(part_data[0]),
                color_idx=color_indeces[ipart - 4 - 1],
                helicity=helicity_conf[ipart - 4 - 1],
                px=float(part_data[1]),
                py=float(part_data[2]),
                pz=float(part_data[3]),
                energy=float(part_data[4]),
            )
            event.add_particle(p)
        return event

    def __iter__(self):
        return self

    def __next__(self):
        # Clear XML iterator
        if self.current:
            self.current[1].clear()

        # Find beginning of new event in XML
        element = next(self.iterator)
        while element[1].tag != "event":
            element = next(self.iterator)
        # Loop over tags in this event
        element = next(self.iterator)
        self.current = element

        return self.unpack_from_iterator()
