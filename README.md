# NLP Political Bias Detection Project

**Repository Navigation**

* **Code:**
    * **Data:** Contains .JSON records used for model training.
    * **RoBERTa_model.py:**  Python script implementing the RoBERTa transformer model for bias detection.
    * **Classical_Models.py:**  Python script for classical machine learning models (Naive Bayes, Logistic Regression).
    * **model.pkl:** Saved, trained RoBERTa model. 
* **Final-Group-Presentation:**
    * **app.py:**  Streamlit application script for an interactive demonstration of the bias detection model.
    * **supporting_images/ **: Images used within the presentation app.
    * **model.pkl:** RoBERTa model utilized in the app.
* **Final-Group-Project-Report.pdf:**  Comprehensive project report outlining methodology, results, and conclusions.

* **[Ani-Meliksetyan-individual-project]:** Individual contributions by [Ani Meliksetyan].
* **[Christopher-Washer-individual-project]:** Individual contributions by [Christopher Washer].
* **[Timur-Abdygulov-individual-project]:** Individual contributions by [Timur Abdygulov].


**Code Execution Order & Descriptions**

1. **Data Preparation (if needed):** If your .JSON data requires further preprocessing, include instructions or reference scripts for this step. 
2. **Model Training:**
   * **RoBERTa_model.py:**  Execute this script to train the RoBERTa transformer model.  Adjust data paths if running locally.
   * **Classical_Models.py:**  Execute this script to train any classical machine learning models used for comparison.
3. **Interactive Demonstration:** 
    * Navigate to the "Final-Group-Presentation" folder.
    * Run `streamlit run app.py` to launch the Streamlit application.(please note that folder contains images for visual accommodation)

