# 🛡️ GBV Case Management Dashboard

## Project Overview
This project, developed as a Senior Year Capstone Project at Parami University, introduces an AI-assisted tool designed to support humanitarian and community workers in managing Gender-Based Violence (GBV) cases. Leveraging Multi-Task Learning (MTL) with a RoBERTa-based model, the dashboard provides real-time classification of GBV incident narratives, including type and severity, and offers AI-generated casework advice.

**Author:** San San Maw
**Class:** Class of 2026, Parami University

## Features
- **GBV Type Classification:** Categorizes incident narratives into distinct GBV types (e.g., Sexual Violence, Physical Violence, Economic Violence, Emotional Violence, Harmful Traditional Practices, Non-GBV).
- **Intensity Scoring:** For certain GBV types, the model provides an intensity level (Low, Medium, High) to help prioritize cases.
- **AI Casework Advisor:** Integrates a Large Language Model (LLM) (via Groq API) to provide structured, trauma-informed guidance based on the incident classification and severity.
- **Human-in-the-Loop Feedback:** Allows caseworkers to provide feedback on model predictions, which can be used to retrain and improve the model over time.
- **Batch Analysis:** Supports uploading CSV/Excel files for bulk classification of incident narratives.

## Getting Started

### Local Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sansanmaw/gbv-classification.git
    cd gbv-classification
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Model Artifacts:** The trained model and label mappings are saved in the `gbv_mtl_roberta_model/` directory. Ensure these are present. If you've trained the model in Google Colab, you would copy them from there.

5.  **Set up API Keys:**
    -   **Groq API Key:** For the AI Casework Advisor, you'll need a Groq API key. Create a `.streamlit/secrets.toml` file in your project root with:
        ```toml
        GROQ_API_KEY="your_groq_api_key_here"
        ```
        *Do not commit this file to public repositories.* For local development, you can create this file directly. For deployment (e.g., Streamlit Cloud), configure secrets directly in the platform's settings.
    -   **Supabase (Optional):** If you plan to use human feedback, configure `SUPABASE_URL` and `SUPABASE_KEY` in `secrets.toml` or directly in your Streamlit Cloud secrets.

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    Your application will open in your web browser.

### Google Colab
This project is developed using Google Colab. You can run the `.ipynb` notebook directly, which handles dependency installation, model training, and artifact saving. The `app.py` is then generated and can be deployed or run within Colab using `streamlit run app.py` (which requires `ngrok` for public access).

## Ethical Considerations
This tool is designed to *assist*, not replace, professional judgment in sensitive GBV cases. AI models can exhibit biases present in their training data, and their predictions should always be reviewed and verified by trained humanitarian and community workers. The AI Casework Advisor provides general guidance and should not be considered a substitute for expert advice or clinical assessment.

## Contact
For any questions or further information, please feel free to reach out:

San San Maw
Email: mawsansan073@gmail.com
