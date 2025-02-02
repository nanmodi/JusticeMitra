# JusticeMitra ⚖️

## Overview
JusticeMitra is an AI-powered legal assistant designed to help women in India understand their legal rights and protections. Using a **Retrieval-Augmented Generation (RAG)** approach, the system processes legal documents and provides accurate, context-aware responses to user queries.

## Features
- 📖 **Legal Query Resolution** – Get detailed responses based on Indian women's rights laws.
- 🔍 **Intelligent Search** – Uses FAISS for efficient document retrieval.
- 🧠 **AI-Powered Responses** – Utilizes ChatGroq for generating legal answers.
- 📂 **PDF Document Processing** – Parses and indexes legal PDFs for reference.
- 🌐 **User-Friendly Web App** – Built with Streamlit for an intuitive interface.
- 📜 **Emergency Contacts** – Displays helpline numbers for immediate assistance.
- 📌 **Search History Sidebar** – Allows users to revisit previous queries.

## How It Works
1. **Upload and Index Legal Documents** – JusticeMitra processes women's rights-related PDFs and builds a searchable FAISS index.
2. **User Query Input** – Users type their legal questions in the web app.
3. **Document Retrieval** – The system fetches relevant legal text from the indexed documents.
4. **AI Response Generation** – ChatGroq processes the retrieved context and provides a well-structured response.
5. **Display Results** – The app presents the response along with emergency legal contacts.

## Built With
- **Languages**: Python
- **Frameworks**: Streamlit, LangChain, FAISS
- **AI Models**: HuggingFace Embeddings, ChatGroq
- **Cloud Services**: Google Cloud / AWS (Optional)
- **APIs**: Groq API
- **Databases/Storage**: Local Storage, Google Drive / AWS S3 (Optional)
- **Tools**: Git, VS Code

## Installation & Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/nanmodi/JusticeMitra.git
   cd JusticeMitra
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root and add:
   ```sh
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application**:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the **JusticeMitra** web app.
2. Enter your legal query in the text box.
3. The AI fetches relevant legal information and provides a response.
4. View **emergency legal contacts** for further assistance.
5. Access previous searches from the sidebar.


![Screenshot 2025-02-01 230045](https://github.com/user-attachments/assets/fd2a5753-d641-48ee-b29b-859c85a04a69)



## Challenges Faced
- Ensuring **accurate legal information retrieval**.
- Handling **large PDF documents** efficiently.
- Optimizing **response relevance** using RAG.
- Improving **user experience** with a modern UI.

## Future Improvements
- 📌 **Multilingual Support** for wider accessibility.
- 📊 **Legal Case Summaries** using advanced NLP.
- 🔗 **Integration with Legal Experts** for personalized guidance.
- ☁️ **Cloud-Based Storage** for real-time document updates.



## License
This project is licensed under the **MIT License**.

## Contact
For questions or collaborations, reach out via **[GitHub Issues](https://github.com/nanmodi/JusticeMitra/issues)**.
