import lamini
import logging
import sqlite3
import pandas as pd
from util.get_schema import get_schema
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()
load_dotenv()


lamini.api_key = os.getenv('LAMINI_API_KEY')
llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")


user = """Who is the highest paid NBA player?"""
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries. Consider the nba_roster table with the following schema:
{get_schema()}
Write a sqlite query to answer the following question. Follow instructions exactly"""
prompt = make_llama_3_prompt(user, system)

print(prompt)
result = llm.generate(prompt, output_type={"Query":"str"})
print(result)

df = pd.read_sql(result['Query'], con=engine)
print(df)