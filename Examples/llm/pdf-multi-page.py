# Split a PDF into individual images and feed to OpenAI API
# Version: 2025-3-3

# Settings
api_key="your_api_key"
pdf_path = "pictures.pdf"        # Replace with your PDF file path
prompt = "Describe this image:"  # Prompt
n_jobs = 2                       # No. of simultaneous calls to model

import pymupdf
import cv2
import numpy as np
import base64
from multiprocessing import Pool
from openai import OpenAI

client = OpenAI(
    base_url = 'https://scrp-chat.econ.cuhk.edu.hk/api',
    api_key=api_key,
)

def send_page_to_model(page_num):
    """
    Function that extract a PDF page, convert to base64 format
    and send to OpenAI API.
    """
    # Extract the page as a Pixmap (image)
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    
    # Convert the Pixmap to a NumPy array (OpenCV format)
    # Ensure the image is in RGB format
    np_array = np.frombuffer(pix.samples, dtype=np.uint8)
    np_array = np_array.reshape(pix.h, pix.w, pix.n)
    
    # Convert the NumPy array to bytes
    _, buffer = cv2.imencode(".png", cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR))
    
    # Convert the bytes to base64 encoding
    base64_bytes = base64.b64encode(buffer).decode("utf-8")
               
    response = client.chat.completions.create(
        model="vision",
        messages=[
            {"role": "user", "content":[ 
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64_bytes}"}}
                ]
            }
        ],
        temperature=0.1,
        )
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # open PDF
    doc = pymupdf.open(pdf_path)
    
    response_list = []
    # Multiprocessing
    with Pool(n_jobs) as p:
        response_list = p.map(send_page_to_model, range(doc.page_count))
        
    print(response_list)