import fitz  

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = "C:\\Users\\chall\\OneDrive\\Desktop\\vs code desktop\\AI research assistant\\data\\1310.4546v1.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Extracted text snippet:\n", extracted_text[:500])
