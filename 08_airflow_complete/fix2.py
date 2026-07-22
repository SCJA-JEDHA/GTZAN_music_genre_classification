import sys

with open('docker-compose.yaml', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'RESULT_BACKEND' in line and '[REDACTED]' in line:
        lines[i] = line.replace('[REDACTED]', 'airflow')
        print(f"Fixed line {i}: {lines[i].strip()}")

with open('docker-compose.yaml', 'w', encoding='utf-8') as f:
    f.writelines(lines)
