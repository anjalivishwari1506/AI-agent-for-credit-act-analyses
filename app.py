import os
import json
from flask import Flask, request, render_template, make_response
from pypdf import PdfReader
from dotenv import load_dotenv
from agent_core import run_agent
import base64
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"ERROR: Could not extract text from PDF. Details: {e}"

@app.route('/', methods=['GET', 'POST'])
def process_pdf():
    report_json = None
    error_message = None
    download_data = None 

    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            error_message = 'No file part in the request.'
        else:
            file = request.files['pdf_file']
            
            if file.filename == '':
                error_message = 'No selected file.'
            elif file and file.filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                try:
                    extracted_text = extract_text_from_pdf(filepath)

                    if extracted_text.startswith("ERROR"):
                        error_message = extracted_text
                    else:
                        final_report = run_agent(extracted_text)
                        
                        if "error" in final_report:
                            error_message = f"AI Agent Error: {final_report['error']}. Check API key or agent_core.py."
                        else:
                           
                            report_json_str = json.dumps(final_report, indent=4)
                            report_json = report_json_str 

                            
                            json_for_b64 = json.dumps(final_report)
                            download_data = base64.b64encode(json_for_b64.encode('utf-8')).decode('utf-8')
                            
                            output_filename = "final_structured_report.json"
                            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                            with open(output_filepath, 'w', encoding='utf-8') as f:
                                f.write(report_json_str)

                except Exception as e:
                    error_message = f"An unexpected error occurred during processing: {e}"
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                error_message = 'Invalid file format. Please upload a PDF file.'
    
    return render_template('index.html', report_json=report_json, error_message=error_message, download_data=download_data)

if __name__ == '__main__':
    app.run(debug=True)