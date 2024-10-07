import requests

API_URL = "http://localhost:8080/generate/"

def send_prompt(prompt):
    try:
        # Correct JSON format with the prompt field
        payload = {
            "prompt": prompt
        }

        # Send POST request to the API
        response = requests.post(API_URL, json=payload)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the response JSON
        result = response.json()

        # Return the generated text
        return result.get("response", "No response generated.")
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

if __name__ == "__main__":
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")

        if prompt.lower() == 'exit':
            print("Exiting...")
            break

        # Send prompt to the AI and get the response
        response = send_prompt(prompt)

        # Display the response from the AI
        print(f"AI Response: {response}\n")
