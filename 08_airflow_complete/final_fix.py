import re

with open('docker-compose.yaml', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find all lines with RESULT_BACKEND and replace [REDACTED] with airflow
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'RESULT_BACKEND' in line:
        # Replace [REDACTED] with the actual password
        lines[i] = line.replace('[REDACTED]', 'airflow')
        print(f"Fixed line {i}: {lines[i]}")

new_content = '\n'.join(lines)

with open('docker-compose.yaml', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("File updated!")
