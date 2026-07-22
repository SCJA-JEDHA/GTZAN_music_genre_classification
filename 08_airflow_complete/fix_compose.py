with open('docker-compose.yaml', 'rb') as f:
    content = f.read()
    
new_content = content.replace(b'[REDACTED]', b'airflow')

with open('docker-compose.yaml', 'wb') as f:
    f.write(new_content)

print("Fixed RESULT_BACKEND password!")
