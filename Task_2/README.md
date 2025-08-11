# Task 2: Flan-T5-large

This is a T5-large model fine-tuned on both SQuAD-v1.1 and the clickbait dataset.

To run this:
1. Activate virtual environment, install requirements, and run `setup.py` in the root folder
2. Make sure the `data` folder exists in the root directory. Then, copy the `.jsonl` data files to the `data` folder
3. Make sure the folder `encoded_data` exists
4. Download SQuAD-v1.1 from [this link](https://www.kaggle.com/datasets/akashdesarda/squad-v11), put it in the `data` folder, and then run `python squad_processing.py` to produce a weighted sample of 20,000 rows.
5. Run `python dataset.py` to produce the tensor datasets
6. Run `python main.py` to begin training. This will save the models in `.safetensors` form to the current folder.
7. Run `python testing.py` to produce a `.csv` file of predictions.