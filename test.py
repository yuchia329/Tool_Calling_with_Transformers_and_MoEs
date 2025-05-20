import json
import ast

# Original input string (you would load this from a file or other source)
raw_string = """{'system': 'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\\n{\\n    "name": "search_books",\\n    "description": "Search for books based on title or author",\\n    "parameters": {\\n        "type": "object",\\n        "properties": {\\n            "query": {\\n                "type": "string",\\n                "description": "The search query"\\n            },\\n            "author": {\\n                "type": "string",\\n                "description": "The author of the book"\\n            }\\n        }\\n    }\\n}\\n\\n{\\n    "name": "calculate_tip",\\n    "description": "Calculate the tip amount for a bill",\\n    "parameters": {\\n        "type": "object",\\n        "properties": {\\n            "bill_amount": {\\n                "type": "number",\\n                "description": "The total bill amount"\\n            },\\n            "tip_percentage": {\\n                "type": "number",\\n                "description": "The percentage of tip to be added"\\n            }\\n        },\\n        "required": [\\n            "bill_amount",\\n            "tip_percentage"\\n        ]\\n    }\\n}\\n\\n', 'chat': 'USER: I am looking for a book but I can\\'t remember the title. The author\\'s name is George Orwell.\\n\\n\\nASSISTANT: <functioncall> {"name": "search_books", "arguments": \\'{"query": "", "author": "George Orwell"}\\'} <|endoftext|>\\n\\n\\nFUNCTION RESPONSE: {"results": [{"title": "1984", "author": "George Orwell"}, {"title": "Animal Farm", "author": "George Orwell"}, {"title": "Down and Out in Paris and London", "author": "George Orwell"}]}\\n\\n\\nASSISTANT: I found several books by George Orwell. Here are a few: "1984", "Animal Farm", and "Down and Out in Paris and London". <|endoftext|>\\n\\n\\nUSER: Oh yes, I was looking for Animal Farm. Thank you!\\n\\n\\nASSISTANT: You\\'re welcome! If you need help with anything else, feel free to ask. <|endoftext|>\\n\\n\\n'}"""

# Step 1: Strip the outer triple quotes
# stripped = raw_string.strip('"""')

# Step 2: Use `ast.literal_eval` to safely parse the string with single quotes into a Python dict
data_dict = ast.literal_eval(raw_string)

# Now you can access the fields
# print(data_dict['system'])  # The system prompt
# print(data_dict['chat'])    # The chat history

print(data_dict['system'])
print('-----')
print(data_dict['chat'])
