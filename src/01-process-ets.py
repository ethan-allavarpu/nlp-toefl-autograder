import os
import pandas as pd
import sys


def join_essay_score(essay_file: str, join_table: pd.DataFrame) -> pd.Series:
    """
    Join essay from .txt file with test taker information and score

    Parameters
    ----------
    essay_file: str
        Path to .txt file which contains participant essay
    join_table: pd.DataFrame
        Table with information about participant demographics and score

    Returns
    -------
    Pandas Series object with ID, prompt, language, response, and score
    """
    with open(essay_file, "r") as f:
        essay = f.read().strip()
    # Extract filename to match join table
    essay_id = essay_file.split("/")[-1]
    essay_dictionary = {
        "essay_id": essay_id,
        "prompt": join_table.loc[essay_id]["Prompt"],
        "language": join_table.loc[essay_id]["Language"],
        "response": essay,
        "score": join_table.loc[essay_id]["Score Level"],
    }
    return pd.Series(essay_dictionary)


def get_all_filepaths(directory: str) -> list:
    """
    Get list of all filepaths for individual ETS data text files

    Parameters
    ----------
    directory: str
        Directory which houses the participant responses

    Returns
    -------
    List with all unique .txt file paths to read
    """
    filepaths = []
    for file in os.listdir(directory):
        if file != ".DS_Store":
            filepaths.append(os.path.join(directory, file))
    return sorted(filepaths)


def write_ets_csv(
    directory: str, output_filepath: str, join_table: pd.DataFrame
) -> int:
    """
    Combine the extracted essay data from all .txt files into a single CSV file
    and log the number of observations present

    Parameters
    ----------
    directory: str
        Data directory housing all .txt files
    output_filepath: str
        Filepath to which to write CSV with tabular data
    join_table: pd.DataFrame
        Table with information about participant demographics and score

    Returns
    -------
    Integer for the number of observations (participants) in the CSV file
    """
    essay_files = get_all_filepaths(directory)
    essay_df = [join_essay_score(file, join_table) for file in essay_files]
    essay_df = pd.DataFrame(essay_df)
    essay_df.to_csv(output_filepath, index=False)
    return int(essay_df.shape[0])


if __name__ == "__main__":
    if (len(sys.argv)) != 4:
        print("Usage:")
        print("  $ python3 src/01-process-ets.py", end=" ")
        print("<ets_dir> <join_tab> <out_path>")
        sys.exit(0)

    ets_dir = os.path.join(os.getcwd(), sys.argv[1])
    join_tab = pd.read_csv(
        os.path.join(os.getcwd(), sys.argv[2]), dtype=str, index_col="Filename"
    )
    output_path = sys.argv[3]

    n_obs = write_ets_csv(ets_dir, output_path, join_tab)
    print(f"CSV file written to {output_path}. Data has {n_obs} observations.")
