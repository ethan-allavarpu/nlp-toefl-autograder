import os
import pandas as pd
import sys

def get_essays_from_file(directory: str, filename: str) -> dict:
    """
    Return text given a directory and filename

    Parameters
    ----------
    directory: str
        Directory which houses the participant response txt files
    filename: str
        Filename of the participant response txt file

    Returns
    -------
    Text of the file
    """
            
    return open(os.path.join(directory, filename), "r").read().strip()

def get_essays_from_code(directory: str, student_id: str) -> list:
    """
    Get map of all essays for an individual from the ICNALE text files

    Parameters
    ----------
    directory: str
        Directory which houses the participant response txt files
    student_id: str
        Participant ID to search for, e.g. "W_CHN_001"

    Returns
    -------
    Map in the form {essay_id: essay_text}
    """
    essay_map = {}
    for file in os.listdir(directory):
        if file[0:6] == student_id[0:6] and file[11:14] == student_id[6:]:
            essay_map[file[6:10]] = open(os.path.join(directory, file), "r").read().strip()
            
    return essay_map


def write_icnale_csv(infosheet_filepath: str, directory: str, output_filepath: str, edited: bool) -> int:
    """
    Combine the extracted essay data from all XML files into a single CSV file
    and log the number of observations present

    Parameters
    ----------
    infosheet_filepath: str
        Filepath to the ICNALE infosheet
    directory: str
        Data directory housing all txt files
    output_filepath: str
        Filepath to which to write CSV with tabular data

    Returns
    -------
    Integer for the number of observations (participants) in the CSV file
    """
    base_df = pd.read_csv(infosheet_filepath)
    base_df = base_df.dropna(thresh=base_df.shape[0]/2, axis=1)

    if not edited:
        base_df["essay_map"] = base_df["Code"].apply(lambda x: get_essays_from_code(directory, x))
        essay_df = pd.json_normalize(base_df["essay_map"]).apply(lambda x: x.str.slice(start=1))
        full_df = pd.concat([base_df.drop(['essay_map'], axis=1), essay_df], axis=1)
        full_df = pd.concat([full_df.drop(["SMK0"], axis=1).rename({"PTJ0": "essay"}, axis=1), full_df.drop(["PTJ0"],axis=1).rename({"SMK0": "essay"}, axis=1)])
    else:
        base_df["essay"] = base_df["File Name"].apply(lambda x: get_essays_from_file(directory, x)).str[1:]
        full_df = base_df.drop(["File Name"], axis=1)

    full_df.to_csv(output_filepath, index=False)

    return int(full_df.shape[0])

if __name__ == "__main__":
    if (len(sys.argv)) != 5:
        print("Usage:")
        print("  $ python3 src/process-icnale.py <infosheet_filepath> <essays_directory> <output_filepath> <edited_essays_bool>")
        sys.exit(0)

    infosheet_filepath = sys.argv[1]
    essays_directory = sys.argv[2]
    output_path = sys.argv[3]
    edited_essays = sys.argv[4].lower() == 'true'
    
    n_obs = write_icnale_csv(infosheet_filepath, essays_directory, output_path, edited_essays)
    print(f"CSV file written to {output_path}. Data has {n_obs} observations.")
    # $ python3 process-icnale.py ../data/icnale/info_written_2.4.csv ../data/icnale/datasets/written_2.4 ../data/icnale-data-written.csv false
    # $ python3 process-icnale.py ../data/icnale/info_edited.csv ../data/icnale/datasets/edited ../data/icnale-data-edited.csv true