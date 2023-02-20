import os
import pandas as pd
import sys

def get_participant_essays(directory: str, student_id: str) -> list:
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


def write_icnale_csv(infosheet_filepath: str, directory: str, output_filepath: str) -> int:
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
    base_df["essay_map"] = base_df["Code"].apply(lambda x: get_participant_essays(directory, x))
    essay_df = pd.json_normalize(base_df["essay_map"]).apply(lambda x: x.str.slice(start=1))
    full_df = pd.concat([base_df.drop(['essay_map'], axis=1), essay_df], axis=1)

    full_df.to_csv(output_filepath, index=False)
    return int(full_df.shape[0])

if __name__ == "__main__":
    if (len(sys.argv)) != 4:
        print("Usage:")
        print("  $ python3 src/process-icnale.py <infosheet_filepath> <essays_directory> <output_filepath>")
        sys.exit(0)

    infosheet_filepath = sys.argv[1]
    essays_directory = sys.argv[2]
    output_path = sys.argv[3]
    
    n_obs = write_icnale_csv(infosheet_filepath, essays_directory, output_path)
    print(f"CSV file written to {output_path}. Data has {n_obs} observations.")
    # $ python3 process-icnale.py ../data/icnale/info_written_2.4.csv ../data/icnale/datasets/written_2.4 ../data/icnale-data.csv