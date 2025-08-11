# Task 1: RoBERTa-large

This is a fine-tuned RoBERTa-large model with an extra fully-connected block to process additional features.

To run this:
1. Activate virtual environment, install requirements, and run `setup.py` in the root folder
2. Make sure the `data` folder exists in the root directory. Then, copy the `.jsonl` data files to the `data` folder
3. Make sure the folder `encoded_data` exists
4. Run `python dataset.py` to produce the tensor datasets
5. Run `python main.py` to begin training. This will save the model in `.pth` form to the `models` folder.
6. Run `python testing.py` to produce a `.jsonl` file of predictions.