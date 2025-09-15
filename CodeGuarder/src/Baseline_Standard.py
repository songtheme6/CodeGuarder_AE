import argparse
import json
import os
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
from copy import deepcopy
from utils import remove_duplicates_preserve_order, read_jsonlines
from typing import List
from tqdm import tqdm

def generate_embeddings(model, texts, batch_size=2):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    return embeddings


def build_vector_database(documents, model):
    embeddings = generate_embeddings(model, documents, batch_size=2)
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip(documents, embeddings)),
        embedding=model.encode
    )
    return vector_db

def load_local_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

def find_similar_code(vector_db, query_text, top_k=3):
    similar_docs = vector_db.similarity_search(query_text, k=top_k)
    return similar_docs

def construct_prompt(examples: List[str], instruction:str):
    prefix = "You are an expert code generation assistant. Your task is to generate code based on the provided instruction and reference examples. The instruction describes the desired functionality of the code. Use the provided code examples as a guide for syntax, structure, and best practices, adapting them as needed to meet the instruction's requirements. Only return the code itself—do not include explanations, preambles, or additional text unless explicitly requested.\n\n---\n\n"
    instruction = f"**Instruction:**\n{instruction}\n\n---\n\n"
    
    knowledge = f"**Reference Code Examples:**\n\n"
    for i, example in enumerate(examples):
        knowledge += f"**Example {i+1}:\n** ```\n{example}\n```\n\n"
    return prefix + instruction + knowledge +  "Generate the code based on the instruction, using the reference examples as the reference if helpful.\n"


def main(ori_instruction_path, kng_base_path, embed_model_path, vector_db_path, output_path, shot_num, language="all"):
    ori_instructions = json.load(open(ori_instruction_path, "r", encoding="utf8"))
    _tmp_knowledge = read_jsonlines(kng_base_path)
    sec_knowledge = [x["fix"] for x in _tmp_knowledge]
    _, model = load_local_model(embed_model_path)
    if os.path.exists(vector_db_path):
        print("Loading vector db from local...")
        vector_db = FAISS.load_local(vector_db_path, model.encode, allow_dangerous_deserialization=True)
    else:
        vector_db = build_vector_database(sec_knowledge, model)
        vector_db.save_local(vector_db_path)
    
    new_instructions = []
    for cur_inst in tqdm(ori_instructions):
        new_inst = deepcopy(cur_inst)
        if language != "all" and cur_inst["language"] != language:
            continue
        query = new_inst["test_case_prompt"].split("\n")[0]
        similar_codes = [x.page_content for x in find_similar_code(vector_db, query, top_k=shot_num)]
        new_inst["test_case_prompt"] =  construct_prompt(remove_duplicates_preserve_order(similar_codes), query) 
        new_inst["similar_codes"] = similar_codes
        new_inst["ori_prompt"] = query
        new_instructions.append(new_inst)
    with open(output_path, "w") as f:
        json.dump(new_instructions, f, indent=4)



def parse_args():
    parser = argparse.ArgumentParser(description="Baseline RAG Security Evaluation Parameters")
    parser.add_argument("--embed_model_path", default="jinaai/jina-embeddings-v3", help="Path to embedding model")
    parser.add_argument("--ori_instruction_path", required=True,
                       help="Path to original instructions")
    parser.add_argument("--kng_base_path", required=True,
                       help="Path to knowledge base")
    parser.add_argument("--language", default="all",
                       help="Language setting (default: all)")
    parser.add_argument("--shot_num", type=int, default=1,
                       help="Number of shots (default: 1)")
    parser.add_argument("--output_path",
                       help="Output file path (optional)")
    parser.add_argument("--vector_db_path",
                       help="Custom vector DB path (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Generate paths if not provided
    if not args.output_path:
        args.output_path = "./datasets/instruct_std.json"
    
    if not args.vector_db_path:
        args.vector_db_path = f"./embeddings/kng_sec_code_{os.path.basename(args.embed_model_path)}.faiss"
    
    main(args.ori_instruction_path,
         args.kng_base_path,
         args.embed_model_path,
         args.vector_db_path,
         args.output_path,
         args.shot_num,
         args.language)