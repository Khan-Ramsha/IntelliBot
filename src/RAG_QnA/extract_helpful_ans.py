def extract_answer(response_text):
    marker = "Helpful Answer:"
    if marker in response_text:
        # Find the position of the marker and extract the text following it
        answer = response_text.split(marker, 1)[1].strip()
        return answer
    return "Helpful Answer not found."
