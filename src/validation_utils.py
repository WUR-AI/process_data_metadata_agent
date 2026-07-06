import os
import sys

import json
import pandas as pd
import requests
from tqdm import tqdm
import time
import duckdb

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy, Faithfulness, AspectCritic, FactualCorrectness, BleuScore, RougeScore, SemanticSimilarity
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure

from dotenv import load_dotenv
load_dotenv()
REPO = os.getenv("REPO_PATH")

METADATA_FIELDS = ['name', 'description', 'keywords', 'license', 
                       'spatial', 'spatialCoverage', 'temporal', 'temporalCoverage', 
                       'fileset', 'recordset']
OTHER_FIELDS = ['id', 'model', 'ds_name']

def get_metadata_files(folder_path):
    """
    Get all metadata files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing metadata files.

    Returns:
        list: A list of paths to the metadata files.
    """
    tmp = folder_path.split('/')[-1].split('_')
    
    ds_name = tmp[0]
    model_name = '_'.join(tmp[1:])
    metadata_files = {x: [] for x in METADATA_FIELDS + OTHER_FIELDS}
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue 

        name_info = filename.rstrip(".json").split("_")
        # assert len(name_info) == 4, f"Unexpected filename format: {filename}"
        id_ = name_info[0]
        metadata_files['id'].append(id_)
        metadata_files['model'].append(model_name)
        metadata_files['ds_name'].append(ds_name)

        with open(os.path.join(folder_path, filename), 'r') as f:
            metadata = json.load(f)
            for field in METADATA_FIELDS:
                metadata_files[field].append(metadata.get(field, None))

    df_metadata = pd.DataFrame(metadata_files)

    return df_metadata

def precheck_validate(df_pred, df_annot):

    cols_drop = [x for x in df_pred.columns if all(df_pred[x].isnull())]
    print(f"Dropping columns from df_pred: {cols_drop}")
    df_pred = df_pred.drop(columns=cols_drop)

    if df_annot['id'].dtype == int:
        df_pred['id'] = df_pred['id'].astype(int)

    assert all(df_pred['id'].isin(df_annot['id'])), "Some ids in df_pred are not present in df_annot"
    df_annot = df_annot[df_annot['id'].isin(df_pred['id'])]
    assert len(df_pred) == len(df_annot), "Mismatch in number of rows between df_pred and df_annot after filtering"

    metadata_fields_present = [x for x in METADATA_FIELDS if x in df_pred.columns]
    assert len(metadata_fields_present) > 0, "No metadata fields found in df_pred"

    expected_shared_cols = metadata_fields_present + ['id']
    assert (set(expected_shared_cols) == set(df_pred.columns).intersection(set(df_annot.columns))), "Mismatch in shared columns between df_pred and df_annot"

    df_merge = pd.merge(df_pred, df_annot, on='id', suffixes=('_pred', '_annot'))
    assert len(df_merge) == len(df_pred), "Mismatch in number of rows after merging df_pred and df_annot"

    return df_merge, df_pred, df_annot, metadata_fields_present

def validate_metadata(df_merge, metadata_fields_present):
    pass
