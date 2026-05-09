import sys

def find_extra_brace(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    stack = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for char in line:
            if char == '{':
                stack.append(i + 1)
            elif char == '}':
                if not stack:
                    print(f"Extra closing brace at line {i + 1}")
                    return
                stack.pop()
    
    if stack:
        print(f"Unclosed opening brace at line {stack[-1]}")
    else:
        print("Braces are balanced")

find_extra_brace('/Users/fahamayoub/Desktop/AthlytIQ/DASHBOARD/index.html')
