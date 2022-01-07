import json
from tqdm import tqdm


def read_json_file(filename):
    """Read content from json file and load

    Args:
        filename: (str) the name of file with relative path

    Return:
        (object): loaded json object
    """
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def read_dataset(dataset_name):
    """Read dataset to dict

    Args:
        dataset_name: (str) the name of the dataset

    Return:
        (dict) the pair of paper id and their value dict
    """
    dataset_dict = {}
    for row in tqdm(read_json_file(f"{dataset_name}.jsonl")):
        dataset_dict[row["paper_id"]] = row
    return dataset_dict


def update_with_paper(origin, dataset):
    """Update dataset introduction with matched origin text

    Args:
        origin: (dict) the original dict
        dataset: (dict) the dataset dict

    Return:
        (dict) the updated dataset dict
    """
    for key in dataset:
        dataset[key]["introduction"] = origin.get(key, {"text": "test"})["text"]

    return dataset


def update_dataset(origin, dataset_name):
    """Update dataset with origin dict

    Args:
        origin: (dict) the original dict
        dataset_name: (text) the name of dataset
    """
    dataset_dict = read_dataset(dataset_name)
    updated_dataset = update_with_paper(origin, dataset_dict)
    with open(f"{dataset_name}_updated.jsonl", "w") as f:
        for datum in tqdm(updated_dataset.values()):
            f.write(f"{json.dumps(datum)}\n")


if __name__ == "__main__":
    origin_dict = read_dataset("../SSN/papers.SSN")
    for dataset_name in ["train", "test", "val"]:
        print (f"Now deal with {dataset_name}")
        update_dataset(origin_dict, dataset_name)
        print (f"Done with {dataset_name}")
