import re

def clean_numbers(text):
    return re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

def preprocessing(documents):
    page_contents = [doc.page_content for doc in documents]

    cleaned_contents = []
    for content in page_contents:
        cleaned_content = content.replace("\n", " ").replace("\t", " ").strip()
        cleaned_content = " ".join(cleaned_content.split())  
        cleaned_contents.append(cleaned_content)


    final_cleaned_content = " ".join(cleaned_contents)
    return final_cleaned_content
