import requests

# File and credentials
filename = "test_lira_kde.csv"
token = "42553895"

# Submit
with open(filename, "rb") as f:
    response = requests.post(
        "http://34.122.51.94:9090/mia",
        files={"file": f},
        headers={"token": token}
    )

# Print the server's evaluation results
print("Submission response:")
print(response.json())
