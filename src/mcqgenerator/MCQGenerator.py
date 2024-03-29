import os
import json
import pandas as pd
import traceback
from dotenv import (
    load_dotenv,
)  # imports key value pairs from .env file and can set them as env variables

# from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.callbacks import get_openai_callback
import PyPDF2


# load environment variables from the .env file
load_dotenv()
key = os.getenv("OPENAIAPIKEY")
# print(key)

llm = ChatOpenAI(api_key=key, model_name="gpt-3.5-turbo", temperature=0.3)
# print(llm)


TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

# RESPONSE FORMAT
RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}


quiz_generation_prompt = PromptTemplate(
    input_variables=[
        "text",
        "number",
        "subject",
        "tone",
        "response_json",
    ],  # vars will be input by user
    template=TEMPLATE,
)

#######################################################

# 2. Use LLMChain to connect LLM and prompt
quiz_chain = LLMChain(
    llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True
)
# quiz_chain

#######################################################

# 3. prompt to evaluate generated quiz

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],  # vars will be input by user
    template=TEMPLATE2,
)


review_chain = LLMChain(
    llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True
)
# review_chain

#######################################################

# 4. Connect both chains using sequentialChain

generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)
# generate_evaluate_chain

#######################################################

# 5. Getting data

file_path = f"/home/zohaib/GenAI/data.txt"
# print(file_path)

with open(file_path, "r") as file:
    TEXT = file.read()
# print(TEXT)
# Serialize the Python dictionary into a JSON-formatted string
print(json.dumps(RESPONSE_JSON))


# 6. Token tracking in Langchain
NUMBER = 5
SUBJECT = "machine learning"
TONE = "simple"

# Anything inside the context manager will get tracked.

with get_openai_callback() as cb:
    response = generate_evaluate_chain(
        {
            "text": TEXT,
            "number": NUMBER,
            "subject": SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON),
        }
    )

    print(cb)


# generate_evaluate_chain:
# print(response)

print(f"Total Tokens:{cb.total_tokens}")
print(f"Prompt Tokens:{cb.prompt_tokens}")
print(f"Completion Tokens:{cb.completion_tokens}")
print(f"Total Cost:{cb.total_cost}")


quiz = response.get("quiz")
quiz = json.loads(quiz)
# quiz

#######################################################
# 6. Creating dataframe
quiz_table_data = []
for key, value in quiz.items():
    mcq = value["mcq"]
    options = " | ".join(
        [
            f"{option}: {option_value}"
            for option, option_value in value["options"].items()
        ]
    )
    correct = value["correct"]
    quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
quiz_table_data


quiz = pd.DataFrame(quiz_table_data)
# print(quiz)

quiz.to_csv("machinelearning.csv", index=False)
