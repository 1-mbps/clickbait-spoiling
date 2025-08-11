# clickbait-spoiling
Final project code for MSE 641 (Text Analytics) at the University of Waterloo

Clickbait is online content designed to entice users to click on it. It often does this by withholding key pieces of information, compelling users to click on it to satisfy their curiosity. In this topic's literature, these pieces of withheld information are called "spoilers".

The Clickbait Challenge at SemEval 2023 involves clickbait spoiling. Task 1 is a classification problem where competitors must use the **title** and **content** of an online article to determine whether its spoiler is:
- a short phrase, ("phrase")
- a longer passage of text ("passage"), or
- multiple nonconsecutive pieces of text ("multi")

Meanwhile, Task 2 involves generating spoilers for internet articles based on their title and content. This is an abstractive/extractive text generation task.

Example:
- Title: "Daniel Craig Was Offered A Staggering Amount Of Money To Carry On Playing Bond"
- Spoiler: "$150 million"
- Spoiler type: phrase

I trained separate models for each task. My model for Task 1 was built around a finetuned RoBERTa-large model, with an additional block of fully-connected linear layers for processing features derived from the dataset. My model for Task 2 was a T5-large model that was first finetuned on SQuAD-v1.1, and then finetuned on the clickbait dataset.

The Task 1 model achieved an F1 score of 0.72080 on the test set. This model placed 7th on the MSE 641 Kaggle competition leaderboard. Meanwhile, the task 2 model achieved a Meteor score of 0.45960. This model placed 4th on the leaderboard and outperformed the baseline model produced by the competition organizers.

Kaggle competitions:
- [Task 1](https://www.kaggle.com/competitions/task-1-clickbait-detection-msci-641-s-25)
- [Task 2](https://www.kaggle.com/competitions/task-2-clickbait-detection-msci-641-s-25)

---

MSE 641 is a graduate course offered by the Faculty of Engineering at the University of Waterloo. I took it in Spring 2025 and was the only undergraduate student enrolled in the class. I worked individually and scored higher than several two-person teams on the Kaggle competitions.