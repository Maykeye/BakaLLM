import sys
from pathlib import Path
import json
from tqdm.auto import tqdm
import re
path = Path(sys.argv[1])
lines = path.read_text().splitlines(True)
path =  path.with_suffix(".jsonl")
out = open(path, "w")

is_first = True
part:list[str] = []

def is_new(line: str):
    if not line.startswith(" = "):
        return False
    return line[3] != '='

re_quote = re.compile('" ([^"]+) "')
re_paren = re.compile("([a-zA-Z]) \\(")

def cleanup(line:str):
    line = line.lstrip().rstrip()
    line = line.replace("= =", "==")
    line = line.replace("= =", "==")
    line = line.replace(" '", "'")
    line = line.replace(" ,", ",")
    line = line.replace(" :", ":")
    line = line.replace(" ;", ";")
    line = line.replace(" .", ".")
    line = line.replace("[ ", "[")
    line = line.replace(" ]", "]")
    line = line.replace("( ", "(")
    line = re.sub(re_paren, "\\1(", line)
    line = line.replace(" )", ")")
    line = line.replace(" / ", "/")
    line = line.replace(" $ ", " $")
    line = line.replace(" & ", "&")
    line = line.replace(" @-@ ", "-")
    line = line.replace(" @,@ ", ",")
    line = line.replace(" @.@ ", ".")
    line = re.sub(re_quote, '"\\1"', line)
    return line

def dump():
    text = "\n".join(part)
    text = json.dumps({"text": text})
    print(text, file=out)

for line in tqdm(lines):
    if is_new(line):        
        if not is_first:
            dump()
        is_first = False
        part = []
    part.append(cleanup(line))

dump()

