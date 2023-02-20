import pandas as pd

REPLACE_DICT = {
    "STUDENT_NAME": "Charlie",
    "Generic_Name": "Charlie",
    "LOCATION_NAME": "Chicago",
    "Generic_Location": "Chicago",
    "Generic_City": "Chicago",
    "TEACHER_NAME": "Mr. Smith",
    "Generic_School": "University of Chicago",
}


if __name__ == "__main__":
    df = pd.read_csv("data/ell/datasets/original_data.csv")
    df['full_text'] = df['full_text'].replace(REPLACE_DICT)
    df.to_csv("data/ell-data.csv", index=False)
