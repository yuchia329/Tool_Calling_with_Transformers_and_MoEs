import ast, json, re
content ="""{"name": "get_random_joke", "arguments": {}}"""
first_quote_pos = content.find("arguments") + len("arguments") + content[content.find("arguments") + len("arguments"):].find("'")
last_quote_pos = last_quote_pos = content.rfind("'")
content_list = list(content)
content_list[first_quote_pos] = "\""
content_list[last_quote_pos] = "\""
content = "".join(content_list)
print(content)
content = content.replace("true", "'True'")
content = content.replace("false", "'False'")
content = ast.literal_eval(content)
arguments = content["arguments"]
if type(arguments) == str:
    arguments = arguments.replace("true", "'True'")
    arguments = arguments.replace("false", "'False'")
    arguments = ast.literal_eval(arguments)
    
content["arguments"] = arguments
content = json.dumps(content)
gold_call = json.loads(re.search(r"\{.*\}", content, re.S).group())