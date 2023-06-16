from pathlib import Path

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def folder_structure(df, dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)