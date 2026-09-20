import json
import os


DATASET_DIR = "datasets/rsvqa"
OUTPUT_FILE = "datasets/rsvqa/training_data.json"


def create_training_data():

    with open(
        os.path.join(DATASET_DIR, "all_questions.json"),
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)["questions"]

    with open(
        os.path.join(DATASET_DIR, "all_answers.json"),
        "r",
        encoding="utf-8"
    ) as file:
        answers = json.load(file)["answers"]

    answers_by_question = {
        item["question_id"]: item["answer"]
        for item in answers
    }

    training_data = []

    for item in questions:

        question_id = item["id"]
        image_id = item["img_id"]

        if question_id not in answers_by_question:
            continue

        training_data.append({
            "image": f"images/Images_LR/{image_id}.tif",
            "question": item["question"],
            "answer": answers_by_question[question_id]
        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            training_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("Training data created successfully!")
    print("Total samples:", len(training_data))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    create_training_data()