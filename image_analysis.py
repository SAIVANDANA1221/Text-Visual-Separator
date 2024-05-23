import cv2
import pytesseract
from pytesseract import Output
import os

def load_img(img_path):
    cap = cv2.VideoCapture(img_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Unable to read video at {img_path}")
    return frame

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged, gray

def extract_text(gray_image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray_image, config=custom_config)
    return text

def segment_visual_elements(edged_image, original_image):
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visual_elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        visual_element = original_image[y:y+h, x:x+w]
        visual_elements.append((x, y, w, h, visual_element))
    return visual_elements

def save_visual_elements(visual_elements, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    element_paths = []
    for i, (x, y, w, h, element) in enumerate(visual_elements):
        element_path = os.path.join(output_dir, f"element_{i}.png")
        cv2.imwrite(element_path, element)
        element_paths.append(element_path)
    return element_paths

def generate_html(text, element_paths):
    html_content = "<html><body>\n"
    html_content += f"<p>{text}</p>\n"
    for element_path in element_paths:
        html_content += f'<img src="{element_path}" />\n'
    html_content += "</body></html>"
    return html_content

def save_html(html_content, output_path):
    with open(output_path, 'w') as html_file:
        html_file.write(html_content)

def main(img_path, output_dir, html_output_path):
    frame = load_img(img_path)
    edged_image, gray_image = preprocess_image(frame)
    text = extract_text(gray_image)
    visual_elements = segment_visual_elements(edged_image, frame)
    element_paths = save_visual_elements(visual_elements, output_dir)
    html_content = generate_html(text, element_paths)
    save_html(html_content, html_output_path)

if __name__ == "__main__":
    img_path = "paste your image path"
    output_dir = "output_elements"
    html_output_path = "output.html"
    main(img_path, output_dir, html_output_path)
