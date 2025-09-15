import re
from io import StringIO
import tokenize
import tempfile
import mistune
import json
import torch
import os
import subprocess
import requests
import sacrebleu
from transformers import AutoTokenizer, AutoModel
from configs import models_config
from openai import OpenAI
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing import List

    
def query(prompt:str, history_msg=[], model="deepseek-chat", uplimit=6000):
    try:
        if model in models_config:
            model_info = models_config[model]
        else:
            raise Exception("model not supported")
        
        client = OpenAI(api_key=model_info["key"], base_url=model_info["base_url"])
        
        if "max_token" in model_info:
            response = client.chat.completions.create(
                model=model,
                messages= history_msg +
                [{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
                max_tokens=model_info["max_token"]
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages= history_msg +
                [{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
            )
        
        history_msg.append({"role": "user", "content": prompt})
        history_msg.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content, history_msg
    
    except Exception as e:
        print(e)
        return "", None
    
def get_splitter(language: str, chunk_size: int, chunk_overlap: int):
    """
    Encapsulates the creation of a RecursiveCharacterTextSplitter for a given language.
    
    Args:
        language (str): The programming language for which the splitter is created.
                        Supported languages include 'python', 'javascript', 'typescript', 
                        'html', 'markdown', 'solidity', 'csharp', 'php', 'haskell', 
                        'cpp', and 'c++'. 'cpp' and 'c++' are treated as the same language.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    
    Returns:
        RecursiveCharacterTextSplitter: A text splitter configured for the specified language.
    
    Raises:
        ValueError: If the specified language is not supported.
    """
    # Normalize the language input to handle both 'cpp' and 'c++'
    if language.lower() in ["cpp", "c++"]:
        language = Language.CPP
    elif language.lower() in ["js", "javascript"]:
        language = Language.JS
    else:
        try:
            language = getattr(Language, language.upper())
        except AttributeError:
            raise ValueError(f"Unsupported language: {language}. Please use one of the supported languages.")

    # Create and return the splitter
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    


def extract_code_blocks(markdown_text):
    class CodeBlockExtractor(mistune.HTMLRenderer):
        def __init__(self):
            super().__init__()
            self.code_blocks = []

        def block_code(self, code, info=None):
            self.code_blocks.append(code)
            return super().block_code(code, info)

    renderer = CodeBlockExtractor()
    markdown = mistune.create_markdown(renderer=renderer)

    markdown(markdown_text)
    return renderer.code_blocks

def extract_content_in_code_blocks(input: str) -> list[str]:
    # Using regular expression to find content between code blocks ```
    return re.findall(r"```(?:[^\n]*)\n(.*?)```", input, re.DOTALL)

def remove_duplicates_preserve_order(lst: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def read_jsonlines(file_path):

    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                data_list.append(data)
    except Exception as e:
        print(f"Error: {e}")
    return data_list


def load_local_model(model_path, device_id=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    if torch.cuda.is_available():
        if device_id is not None:

            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    
    return tokenizer, model

def download_raw_github_file(github_url, proxies=None):
    """
    使用 raw.githubusercontent.com 从 GitHub 下载文件内容。

    参数：
        github_url (str): GitHub 文件的 URL。

    返回：
        str: 文件内容的字符串，如果下载失败则返回 None。
    """
    try:
        # 将 github.com URL 转换为 raw.githubusercontent.com URL
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com", 1).replace("/blob/", "/")

        # 发送 GET 请求
        if proxies is not None:
            response = requests.get(raw_url, proxies=proxies)
        else:
            response = requests.get(raw_url)
        response.raise_for_status()  # 检查请求是否成功

        # 返回文件内容
        return response.text

    except requests.exceptions.RequestException as e:
        print(f"发生错误：{e}")
        return None                 


def extract_functions_from_c_code(code, filename="temp.c"):
    def filter(function_string):
        if function_string == "":
            return False
        if "{" not in function_string:
            return False
        stack = []
        for char in function_string:
            if char == '{':
                stack.append(char)
            elif char == '}':
                if not stack:
                    return False
                stack.pop()

        return not stack 

    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["clang", "-Xclang", "-ast-dump", "-fsyntax-only", "-Wno-everything", "-ferror-limit=0", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        function_pattern = re.compile(
            r"FunctionDecl.*<(?P<file>.*):(?P<start_line>\d+):\d+, line:(?P<end_line>\d+):\d+>.* (?P<name>\w+)"
        )
        functions = []
        for line in result.stdout.splitlines():
            match = function_pattern.search(line)
            if match:
                file_path = match.group("file")
                start_line = int(match.group("start_line"))
                end_line = int(match.group("end_line"))
                function_name = match.group("name")

                if os.path.basename(file_path) == os.path.basename(temp_file_path) or file_path == "line":
                    functions.append({
                        "name": function_name,
                        "start_line": start_line,
                        "end_line": end_line
                    })

        code_lines = code.splitlines()
        function_contents = []
        for func in functions:
            start = func["start_line"] - 1  
            end = func["end_line"]  
            function_body = "\n".join(code_lines[start:end])
            function_contents.append(function_body)

        return [x for x in function_contents if filter(x)]
    
    except Exception as e:
        print(f"error: {e}")
        return []

def extract_js_functions(file_path_or_content, is_content=True):
    try:
      
        if is_content:
            content = file_path_or_content
        else:
            with open(file_path_or_content, 'r', encoding='utf-8') as file:
                content = file.read()
        
      
        patterns = [
            
            r'(?<![:.\w])function\s+([a-zA-Z_]\w*)\s*\((.*?)\)\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',
            
            r'(?:const|let|var)\s+([a-zA-Z_]\w*)\s*=\s*\((.*?)\)\s*=>\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',

            r'(?:const|let|var)\s+([a-zA-Z_]\w*)\s*=\s*function\s*\((.*?)\)\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',

            r'(?:this|[a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\s*=\s*function\s*\((.*?)\)\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',

            r'(?<![\w.])([a-zA-Z_]\w*)\s*:\s*function\s*\((.*?)\)\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}(?![^,])'
        ]
        
        functions = []
        
      
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                functions.append(match.group(0))
        
        return functions
    
    except FileNotFoundError:
        return []
    except Exception as e:
        return []

def compute_bleu_score(hyp: str, ref: str) -> float:
    """Compute BLEU score between two strings using SacreBleu."""

    return sacrebleu.corpus_bleu(
        [hyp],
        [[ref]],
        smooth_method="exp",
        force=False,
        lowercase=False,
        use_effective_order=False,
    ).score
