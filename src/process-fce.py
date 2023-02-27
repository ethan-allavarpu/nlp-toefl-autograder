import os
import pandas as pd
import re
import sys


def clean_response_text(marked_essay) -> str:
    """
    Take response text and remove error tag and corrections

    Parameters
    ----------
    marked_essay: str
        Essay response with error tagging and corrections

    Returns
    -------
    Original essay response, with error tags and corrections removed
    """
    marked_essay = re.sub("(?<=<c>).*?(?=</c>)", "", marked_essay)
    marked_essay = re.sub("</p> <p>", "\n", marked_essay)
    marked_essay = re.sub("<.*?>", "", marked_essay)
    return re.sub(" +", " ", marked_essay).strip()


def extract_essay_data(filepath: str) -> pd.Series:
    """
    Take XML file and return processed data as a row observation

    Parameters
    ----------
    filepath: str
        Path to XML file which contains participant essays

    Returns
    -------
    Pandas Series object with ID, language, age, scores, and text for two essays
    """
    with open(filepath, "r") as f:
        xml_data = f.read()
    str_xml = " ".join([line.strip() for line in xml_data.split("\n")])
    sortkey = re.findall('(?<=<head sortkey=").*?(?=">)', str_xml)
    language = re.findall("(?<=<language>).*?(?=</language>)", str_xml)
    age_grp = re.findall("(?<=<age>).*?(?=</age>)", str_xml)
    if len(age_grp) == 0:
        age_grp = [None]
    overall_score = re.findall("(?<=<score>).*?(?=</score>)", str_xml)
    ans1 = re.findall("(?<=<answer1>).*?(?=</answer1>)", str_xml)[0]
    essay_1 = re.findall("(?<=<coded_answer>).*?(?=</coded_answer>)", ans1)
    response_1 = [clean_response_text(essay_1[0])]
    q1 = re.findall("(?<=<question_number>).*?(?=</question_number>)", ans1)
    exam_score1 = re.findall("(?<=<exam_score>).*?(?=</exam_score>)", ans1)
    if len(exam_score1) == 0:
        exam_score1 = [None]
    ans2 = re.findall("(?<=<answer2>).*?(?=</answer2>)", str_xml)
    # Second essay not always present
    if len(ans2) > 0:
        ans2 = ans2[0]
        essay_2 = re.findall("(?<=<coded_answer>).*?(?=</coded_answer>)", ans2)
        response_2 = [clean_response_text(essay_2[0])]
        q2 = re.findall("(?<=<question_number>).*?(?=</question_number>)", ans2)
        exam_score2 = re.findall("(?<=<exam_score>).*?(?=</exam_score>)", ans2)
        if len(exam_score2) == 0:
            exam_score2 = [None]
    else:
        q2 = [None]
        exam_score2 = [None]
        response_2 = [None]
    # Validate proper extraction
    assert len(sortkey) == 1
    assert len(language) == 1
    assert len(age_grp) == 1
    assert len(overall_score) == 1
    assert len(q1) == 1
    assert len(exam_score1) == 1
    assert len(response_1) == 1
    assert len(q2) == 1
    assert len(exam_score2) == 1
    assert len(response_2) == 1
    # Create dictionary to convert to pandas Series object
    data_dictionary = {
        "sortkey": sortkey[0],
        "language": language[0],
        "age": age_grp[0],
        "overall_score": overall_score[0],
        "response_1_q": q1[0],
        "response_1_score": exam_score1[0],
        "response_1_essay": response_1[0],
        "response_2_q": q2[0],
        "response_2_score": exam_score2[0],
        "response_2_essay": response_2[0],
    }
    return pd.Series(data_dictionary)


def get_all_filepaths(directory: str) -> list:
    """
    Get list of all filepaths for individual FCE data XML files

    Parameters
    ----------
    directory: str
        Directory which houses the participant responses

    Returns
    -------
    List with all unique XML file paths to read
    """
    filepaths = []
    for subdir in os.listdir(directory):
        if re.match("[0-9]", subdir):
            for file in os.listdir(os.path.join(directory, subdir)):
                filepaths.append(os.path.join(directory, subdir, file))
    return sorted(filepaths)


def write_fce_csv(directory: str, output_filepath: str) -> int:
    """
    Combine the extracted essay data from all XML files into a single CSV file
    and log the number of observations present

    Parameters
    ----------
    directory: str
        Data directory housing all XML files
    output_filepath: str
        Filepath to which to write CSV with tabular data

    Returns
    -------
    Integer for the number of observations (participants) in the CSV file
    """
    essay_files = get_all_filepaths(directory)
    essay_df = pd.DataFrame([extract_essay_data(file) for file in essay_files])
    essay_df.to_csv(output_filepath, index=False)
    essay_df.dropna().reset_index(drop=True).to_csv(
        output_filepath.replace(".csv", "-full-obs.csv"), index=False
    )
    essay_df.dropna().loc[
        :, ["sortkey", "response_1_essay", "response_2_essay", "overall_score"]
    ].melt(id_vars=["sortkey", "overall_score"]).drop(columns="variable").rename(
        columns={"value": "essay"}
    ).sort_values("sortkey").to_csv(
        output_filepath.replace(".csv", "-input-format.csv"), index=False
    )
    return int(essay_df.shape[0])


if __name__ == "__main__":
    if (len(sys.argv)) != 3:
        print("Usage:")
        print("  $ python3 src/process-fce.py <fce_dir> <output_filepath>")
        sys.exit(0)

    data_directory = os.path.join(os.getcwd(), sys.argv[1])
    output_path = sys.argv[2]

    n_obs = write_fce_csv(data_directory, output_path)
    print(f"CSV file written to {output_path}. Data has {n_obs} observations.")
