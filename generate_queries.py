import lamini
import logging
import random
from typing import AsyncIterator, Iterator, Union
import sqlite3
import copy
from tqdm import tqdm
import pandas as pd
import jsonlines
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from util.get_schema import get_schema
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()
load_dotenv()

lamini.api_key = os.getenv("LAMINI_API_KEY")

class Args:
    def __init__(self, 
                 max_examples=100, 
                 sql_model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                 gold_file_name="gold-test-set.jsonl",
                 training_file_name="generated_queries_large_filtered.jsonl",
                 num_to_generate=10):
        self.sql_model_name = sql_model_name
        self.max_examples = max_examples
        self.gold_file_name = gold_file_name
        self.training_file_name = training_file_name
        self.num_to_generate = num_to_generate


system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += (
    "Consider a table called 'nba_roster' with the following schema (columns)\n"
)
system += get_schema()
system += "Consider the following questions, and queries used to answer them:\n"

question = """What is the median weight in the NBA?"""
sql = "select CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) as percentile from nba_roster order by percentile limit 1 offset (select count(*) from nba_roster)/2;"

system += "Question: " + question + "\n"
system += "Query: " + sql + "\n"

user = "Write two queries that are similar but different to those above.\n"
user += "Format the queries as a JSON object, i.e.\n"
user += '{ "explanation": str, "sql_query_1" : str, "sql_query_2": str }.\n'

user += "First write an explanation of why you decided to write these new queries in about 3-5 sentences, \
    then write valid sqlite SQL queries for each of the 2 new queries. \
    Make sure each query is complete and ends with a ;\n"

prompt = make_llama_3_prompt(user, system)

llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
result = llm.generate(prompt, output_type={ "explanation": "str", "sql_query_1" : "str", "sql_query_2": "str" }, max_new_tokens=200)
print(result)
