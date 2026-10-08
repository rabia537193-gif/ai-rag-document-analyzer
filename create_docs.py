import os

# Sample docs directory path
docs_dir = "./sample_docs"
os.makedirs(docs_dir, exist_ok=True)

# 5 Sample Documents with content
documents_data = {
    "doc1.txt": """Company Remote Work Policy:
All employees working remotely must register their primary devices with the IT Department before December 1st.
Employees must use two-factor authentication (2FA) when accessing company portals.
Unregistered devices will be blocked from accessing internal systems automatically after the deadline.""",

    "doc2.txt": """Technical Setup and Deployment Guidelines:
The production server requires Python 3.10+ and a minimum of 4GB RAM.
Environment variables such as API keys and database credentials must be loaded via .env files.
Never hardcode sensitive credentials directly inside the codebase or commit them to public version control repositories.""",

    "doc3.txt": """Human Resources & Leave Guidelines:
Annual leave requests must be submitted at least 14 days prior through the HR portal.
Employees are entitled to 20 days of paid leave per calendar year.
Sick leaves exceeding two consecutive working days require an official medical certificate from a registered doctor.""",

    "doc4.txt": """Project AI Knowledge Assistant Overview:
The AI Knowledge Assistant uses Retrieval-Augmented Generation (RAG) to answer user questions based on custom documents.
It utilizes sentence-transformer embeddings to convert document chunks into numerical vectors.
These vectors are stored in ChromaDB to perform fast semantic similarity searches.""",

    "doc5.txt": """IT Support & Security FAQs:
Question: How do I reset my account password?
Answer: Navigate to the self-service portal at portal.company.com and click 'Reset Password'.
Question: What are IT support hours?
Answer: IT Support is available Monday to Friday from 9:00 AM to 5:00 PM PKT."""
}

# Write text into all 5 files
for file_name, content in documents_data.items():
    file_path = os.path.join(docs_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Created/Updated: {file_name}")

print("\nAll 5 documents have been successfully populated with text!")