import os
import json
import pandas as pd
import traceback
import streamlit as st
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

from langchain_community.callbacks import get_openai_callback


with open("/home/zohaib/GenAI/Response.json", "r") as f:
    RESPONSE_JSON = json.load(f)
    # print(RESPONSE_JSON)


# creating a title for page
st.title("MCQ Creator application with langchain")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a pdf or txt")
    # input fields
    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Subject", max_chars=20)
    tone = st.text_input("Complexity level", max_chars=20, placeholder="simple")

    # button
    button = st.form_submit_button("Create MCQ")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": tone,
                            "response_json": json.dumps(RESPONSE_JSON),
                        }
                    )

                # st.write(response)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)

                st.error("Error")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    # extract quiz data from the response
                    quiz = response.get("quiz", None)
                    print(quiz)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # display review in text as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data ")

                else:
                    st.write(response)
