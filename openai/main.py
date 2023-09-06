from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
from dotenv import load_dotenv
import utils
import pandas as pd
import os
import openai
import json
import streamlit as st

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
# Configure OpenAI API
openai.api_type = "azure"
#openai.api_version = "2023-03-15-preview" 
OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

AAD_TENANT_ID = os.getenv("AAD_TENANT_ID")
KUSTO_CLUSTER = os.getenv("KUSTO_CLUSTER")
KUSTO_DATABASE = os.getenv("KUSTO_DATABASE")
KUSTO_TABLE = os.getenv("KUSTO_TABLE")
KUSTO_MANAGED_IDENTITY_APP_ID = os.getenv("KUSTO_MANAGED_IDENTITY_APP_ID")
KUSTO_MANAGED_IDENTITY_SECRET = os.getenv("KUSTO_MANAGED_IDENTITY_SECRET")

# Connect to adx using AAD app registration
cluster = KUSTO_CLUSTER
kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, KUSTO_MANAGED_IDENTITY_APP_ID, KUSTO_MANAGED_IDENTITY_SECRET,  AAD_TENANT_ID)
client = KustoClient(kcsb)

# Test the connection to kusto works - sample query to get the top 10 results from the sales table
#query = "sales | take 10"
query = "MediaCreationService | take 10"

#response = client.execute("AzureOpenAiTest", query)
response = client.execute("WDGES", query)
for row in response.primary_results[0]:
    #print("Item:{}".format(row["item_description"]))
    print("Item:{}".format(row["TaskName"]))

# Create a prompt for the model to generate the KQL query
kusto_query_template_prefix = """
<|im_start|>system
I have an Azure Data Explorer (Kusto) table containing the following columns: 
TimeStamp, ID, JobState, BuildCategory, Branch, BuildName, ArtifactName, FailureMessage, Fingerprint

Each row represents some event associated with a "job". The ID column identifies a unique  job. The TimeStamp column says when the given event occurred. 

Here's an example prompt an answer:

Prompt: How many jobs ran in the last week

Answer: T 
| where TimeStamp > ago(7d)
| summarize dcount(ID)

Write an KQL query based on the user input below. Answer in a concise manner. Answer only with the KQL query where the table name is MediaJobs, no extra text.

user input: 
"""
template_sufix = "<|im_end|>\n<|im_start|>assistant"

# Define functions to call the OpenAI API and run KQL query
def call_openai(template_prefix, text):
    prompt = template_prefix + text + template_sufix
    response = openai.Completion.create(
        engine=utils.OPENAI_DEPLOYMENT_NAME,
        prompt=prompt,
        temperature=0,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"])
    response = response['choices'][0]['text']
    response = utils.remove_chars("\n", response)
    response=utils.start_after_string("Answer:", response)
    response=utils.remove_tail_tags("<|im_end|>", response)
    return response

def call_openai_kql_response(text):
    response = call_openai(kusto_query_template_prefix, text)
    #query = response.replace("T", "sales")
    #query = response.replace("T", "MediaJobs")
    query = response
    query = query.replace("```", "")
    st.write(query)
    #response = client.execute("AzureOpenAiTest", query)
    response = client.execute("WDGES", query)
    df = dataframe_from_result_table(response.primary_results[0])
    return df

# Test the model with a sample question - aggregate the retail sales by month for the year 2020
#df = call_openai_kql_response("I would like to get the total retail sales for 2020 by months")
def run_query(question):
    #df = call_openai_kql_response("What year had the max retail sales and what were those sales")
    df = call_openai_kql_response(question)
    return df

#result = run_query("How many jobs were there in the last 3 days?")    

st.title('i\'m the media kusto table, ask me what i know')

question = st.text_input("enter your question:", "what are the columns?")

if st.button('enter'):
    result = run_query(question)
    #st.write(f"Result: {result}")
    st.dataframe(result)
