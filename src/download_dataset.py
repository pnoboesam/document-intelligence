from datasets import load_dataset


def main():
    dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")

    print(dataset)

    print("\nTrain:")
    print(dataset["train"])

    print("\nTest:")
    print(dataset["test"])

    print("\nFirst example:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()