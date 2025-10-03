# Split a PDF into individual images and feed to OpenAI API
# This version first convert the PDF into png image files,
# before reading and feeding the images to the OpenAI API
# Version: 2025-3-10

# Stage 1 settings
pdf_path = "pictures.pdf"        # Replace with your PDF file path

# Stage 2 Settings
model = "vision"
api_key="your_api_key"
api_url = 'https://scrp-chat.econ.cuhk.edu.hk/api'
output_prefix = "pic-description"
output_suffix = ".csv"
n_jobs = 2                       # No. of simultaneous calls to model
prompt = "Describe this image:"  # Prompt
temperature = 0                  # [0,1]. 0 means no randomness.

import pymupdf
import cv2
import numpy as np
import base64
from multiprocessing import Pool
from openai import OpenAI

def pdf_to_images(pdf_path):
    """
    Extract each page from the PDF, convert to RGB image, and save as PNG.
    """
    # Open the PDF document
    with pymupdf.open(pdf_path) as doc:
        # Iterate through each page
        for page_num in range(doc.page_count):
            # Extract the page as a Pixmap (image)
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            
            # Convert the Pixmap to a NumPy array (OpenCV format)
            # Ensure the image is in RGB format
            np_array = np.frombuffer(pix.samples, dtype=np.uint8)
            np_array = np_array.reshape(pix.h, pix.w, pix.n)

            #np_array = np_array[:round(pix.h/5)]
            
            # Save the image as PNG
            output_filename = f"page_{page_num}.png"
            cv2.imwrite(output_filename, cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR))
            print(f"Saved page {page_num} as {output_filename}")

        return doc.page_count

client = OpenAI(
    base_url = api_url,
    api_key=api_key,
)

def send_page_to_model(page_num):
    """
    Function that extract a PDF page, convert to base64 format
    and send to OpenAI API.
    """

    # Load the image
    img = cv2.imread('page_' + str(page_num) + '.png')

    # Crop the image
    #height, width, channels = img.shape
    #mid_height = height // 2
    #img = img[:mid_height, :]
    
    # Convert the NumPy array to bytes
    _, buffer = cv2.imencode(".png", img)
    
    # Convert the bytes to base64 encoding
    base64_bytes = base64.b64encode(buffer).decode("utf-8")
               
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content":[ 
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64_bytes}"}}
                ]
            }
        ],
        temperature=temperature,
        )
    
    output_text = response.choices[0].message.content
    output_filename = output_prefix + "_" + str(page_num) + output_suffix
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(output_text)

    return output_text

# Example usage
if __name__ == "__main__":
    page_count = pdf_to_images(pdf_path)

    # Simultaneous call to OpenAI API with multiprocessing
    print(f"Sending data to model: {model} ...")
    with Pool(n_jobs) as p:
        response_list = p.map(send_page_to_model, range(page_count))
    print("Done.")
    