
# 🧠 AI Agent for Credit Act Analysis

An **AI-powered legal document analysis system** built using **Flask** and **Gemini API**, designed to analyze **Credit Act–related legal documents**. The system accepts **PDF legal documents** as input and returns a **clear, structured summary** in **simple language** using **key–value pair format** for easy understanding.

This project focuses on making complex legal documents **accessible, accurate, and user-friendly** with the help of modern AI techniques.

---

## 🚀 Features

* 📄 Upload **legal documents in PDF format**
* 🤖 AI-powered analysis using **Google Gemini API**
* 🧩 Extracts **important legal clauses and sections**
* 📝 Generates:

  * Simple language summary
  * Structured **key–value pair output**
* ⚖️ Designed specifically for **Credit Act & financial legal documents**
* 🌐 Uses **web tools + document context** for better understanding
* ⚡ Fast and responsive Flask backend

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **AI Model:** Google Gemini API
* **Document Handling:** PDF parsing
* **AI Approach:** Context-aware document understanding
* **Frontend:** Simple & clean UI (Flask templates)

---

## 🧠 How It Works

1. User uploads a **Credit Act legal PDF**
2. PDF content is extracted and cleaned
3. Extracted text is passed to the **Gemini AI model**
4. AI analyzes the legal content and:

   * Identifies key legal information
   * Converts complex legal language into simple terms
   * Outputs data in **structured key–value format**
5. User receives a **summarized and easy-to-read legal report**

---

## 📂 Project Structure

```
AI-agent-for-credit-act-analyses/
│
├── app.py                  # Main Flask application
├── templates/              # HTML templates
├── static/                 # CSS / JS files
├── utils/                  # PDF processing & helper functions
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🔑 Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anjalivishwari1506/AI-agent-for-credit-act-analyses.git
cd AI-agent-for-credit-act-analyses
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Gemini API Key

Create a `.env` file and add:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 📌 Use Cases

* Legal document summarization
* Credit Act compliance analysis
* Financial and legal research assistance
* Law students & legal professionals
* Simplifying complex legal language for non-technical users

---

## 🔒 Disclaimer

This project is intended for **educational and informational purposes only**.
It **does not replace professional legal advice**.

---

## 🌟 Future Enhancements

* Support for multiple legal domains
* Downloadable summary reports
* Multi-language legal summaries
* Enhanced UI with dashboards

---

## 👩‍💻 Author

**Anjali Vishwari**
📌 AI | Data Science | Backend Development
🔗 GitHub: [anjalivishwari1506](https://github.com/anjalivishwari1506)


