#pip install pandasai pandasai[excel] pandasai[connectors] tiktoken
import os
import csv
import matplotlib
import tiktoken
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.responses.response_parser import ResponseParser



class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return


def setup():
    st.header("Chat with your small and large datasets!", anchor=False, divider="red")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def get_tasks():
    st.sidebar.header("Select a task", divider="rainbow")
    task = st.sidebar.radio("Choose one:",
                            ("Load from local drive, <200MB",
                             "Load from local drive, 200MB+")
                            )
    return task


def get_llm():
    st.sidebar.header("Select a LLM", divider='rainbow')
    llm = st.sidebar.radio("llm:",
                           ("OpenAI")
                           )
    return llm



def calculate_cost(df):  
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")  
    cost = 0.0005  

    strings = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()  
    token_count = [len(encoding.encode(rows)) for rows in strings]  

    total_tokens = sum(token_count)     
    st.write('Tokens:', total_tokens)  
    st.write('Cost:', total_tokens * cost / 1000)


def main():
    """1. setup page
       2. setup options - tasks: load or retrieve, models: bamboo, openai, google
       3. yfinance
    """
    
    setup()
    task = get_tasks()
    
    if task == "Load from local drive, <200MB":
        dataset = st.file_uploader("Upload your csv or xlsx file", type=['csv','xlsx'])
        if not dataset: st.stop()
        df = pd.read_csv(dataset, low_memory=False)
        calculate_cost(df)
        st.write("Data Preview:")
        st.dataframe(df.head())
        col_desc = st.radio("Do you want to provide column descriptors?",
                            ("Yes",
                             "No")
                            )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"
        
        if addon:
            llm = get_llm()
            if llm == "PandasAI":
                connector = PandasConnector({"original_df": df}, field_descriptions=addon)
                sdf = SmartDataframe(connector, {"enable_cache": False})
                prompt1 = st.text_input("Enter your question/prompt.")
                if not prompt1: st.stop()
                response = sdf.chat(prompt1)
                st.write("Response")
                st.write(response)
                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)
                
            elif llm == "OpenAI":
                llm = OpenAI(api_token=OPENAI_API_KEY)
                connector = PandasConnector({"original_df": df}, field_descriptions=addon)
                sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm, "conversational": False, 
                                                        "response_parser": OutputParser})
                prompt2 = st.text_input("Enter your question/prompt.")
                if not prompt2: st.stop()
                st.write("Response")
                with get_openai_callback() as cb:
                    response2 = sdf.chat(prompt2)
                    st.divider()
                    st.write("🧞‍♂️ Under the hood, the code that was executed:")
                    st.code(sdf.last_code_executed)
                    st.divider()
                    st.write("💰 Tokens used and your cost:")
                    st.write(cb)
                    
            
                    
                
    if task == "Load from local drive, 200MB+":
        filename = st.text_input("Enter your file path including filename, e.g. /users/xyz/abc.csv, .csv files only")
        if not filename:st.stop()
        df_large = pd.read_csv(filename, low_memory=False)
        st.write("Data Preview:")
        st.dataframe(df_large.head())
        col_desc = st.radio("Do you want to provide column descriptors?",
                            ("Yes",
                             "No")
                            )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"
        
        if addon:
            llm = OpenAI(api_token=OPENAI_API_KEY)
            connector = PandasConnector({"original_df": df_large}, field_descriptions=addon)
            sdf = SmartDataframe(connector, {"enable_cache": False}, config={"llm": llm, "conversational": False, 
                                                    "response_parser": OutputParser})
            prompt6 = st.text_input("Enter your question/prompt.")
            if not prompt6: st.stop()
            st.write("Response")
            with get_openai_callback() as cb:
                response6 = sdf.chat(prompt6)
                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)
                st.divider()
                st.write("💰 Tokens used and your cost:")
                st.write(cb)
        

if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    PANDASAI_API_KEY = os.environ.get('PANDASAI_API_KEY')
    projectid = os.environ.get('GOOG_PROJECT')
    matplotlib.use("Agg", force=True)
    main()
