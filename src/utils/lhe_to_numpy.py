import gzip
import shutil
import os
import numpy as np
from lhereader import LHEReader

def lhe_to_array(dir: str):
    """
    LHE to numpy array reader
    """
    reader = LHEReader(dir)
    events = []
    for i, event in enumerate(reader):
        # particles_out = filter(lambda x: x.status == 1, event.particles)
        momenta = []
        for particle in event.particles:
            mom = np.array([particle.pdgid, particle.color_idx, particle.helicity, particle.energy, particle.px, particle.py, particle.pz])
            momenta.append(mom)
        momenta = np.hstack(momenta)
        # append as last element the LC_to_FC_factor
        # new_values = [event.amp_times_weight, event.A_LC, event.A_NLC, event.A_FC, event.weight,
        #       event.r_LC_to_FC, event.r_LC_to_NLC, event.weight_LC, event.weight_NLC, event.weight_FC]
        new_values = [event.A_LC, event.A_NLC, event.A_FC]
        momenta = np.append(momenta, new_values, axis=0)

        # momenta = np.append(momenta, event.LC)
        # momenta = np.append(momenta, event.FC)
        events.append(momenta)
        if (i+1) % 10_000 == 0:
            print(f"Read {i+1} events, total shape ({len(events)}, {momenta.shape})")
        # if i==9:
        #     print(f"Read {i+1} events, stopping early for testing.")
        #     break
    events = np.stack(events)
    return events


    

def concat_lhe_across_seeds(
    base_filename: str,
    seed_start: int = 101,
    seed_end: int = 110,
    input_root: str = "/remote/gpu02/marino/data/gg_ddbarng/seeds/",
    output_dir: str = "/remote/gpu02/marino/data/gg_ddbarng/"
) -> str:
    """
    Iterate over seed directories, read each LHE file with the given base filename,
    concatenate the arrays, and save to a single .npy file.

    Parameters
    ----------
    base_filename : str
        The LHE filename (e.g., "events_6_2_21_21_21_21_21_21_1_2_3_4_5_6.lhe.rwgt").
    seed_start : int
        Starting seed number (inclusive).
    seed_end : int
        Ending seed number (inclusive).
    input_root : str
        Root directory containing seed subfolders (default: "data/gg_ng").
    output_dir : str
        Directory where the concatenated .npy file will be saved (default: "data/gg_ng/large").

    Returns
    -------
    output_path : str
        Path to the saved .npy file.
    """
    base_filename+= ".lhe.rwgt"  # Append the file extension
    arrays = []
    total_events = 0
    for seed in range(seed_start, seed_end + 1):
        seed_dir = os.path.join(input_root, f"random_number_seed_{seed}")
        file_path = os.path.join(seed_dir, base_filename)
        # find if there are filenames with .gz
        if not os.path.exists(file_path):
            if os.path.exists(file_path + ".gz"):
                file_path = file_path + ".gz"
            else:
                print(f"File not found: {file_path} (or +.gz)")
                continue
        else:
            pass
        if ".gz" in file_path:
            file_pathgz = file_path
            file_path = file_pathgz.rstrip('.gz')
            with gzip.open(file_pathgz, 'rb') as f_in, open(file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        arr = lhe_to_array(file_path)
        if os.path.exists(file_path + ".gz"):
            os.remove(file_path)
            print(f"Unpacked: {file_pathgz} and removed: {file_path}")
        arrays.append(arr)
        total_events += arr.shape[0]
        print(f"Completed reading seed {seed}. Cumulative events: {total_events}")

    if not arrays:
        raise RuntimeError("No LHE files were read; please check seed range and filenames.")

    concatenated = np.concatenate(arrays, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, base_filename + ".npy")

    np.save(output_path, concatenated)
    print(f"Saved concatenated events to {output_path} (total events: {concatenated.shape[0]})")

    return output_path

# single file usage example
# arr = lhe_to_array("/Users/jamarino/Documents/Heidelberg/Work/MadRecolor/madrecolor/data/gg_ng/events_5g_1M_ext.comb")
# output_dir = "/Users/jamarino/Documents/Heidelberg/Work/MadRecolor/madrecolor/data/gg_ng"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, "events_5g_1M_ext.comb" + ".npy")
# np.save(output_path, arr)

# multiple files usage example
for base_filename in ["events_6_2_21_21_1_-1_21_21_3_1_2_5_6_4",
    "events_7_2_21_21_1_-1_21_21_21_3_1_2_5_6_7_4",
    "events_8_2_21_21_1_-1_21_21_21_21_3_1_2_5_6_7_8_4",
    "events_9_2_21_21_1_-1_21_21_21_21_21_3_1_2_5_6_7_8_9_4"

]:
    concat_lhe_across_seeds(
        base_filename=base_filename
    )