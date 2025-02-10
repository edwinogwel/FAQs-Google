import requests

url = "http://127.0.0.1:5000/chat"
data = {"message": "How do I reset my Google password?"}
response = requests.post(url, json=data)

print(response.json())