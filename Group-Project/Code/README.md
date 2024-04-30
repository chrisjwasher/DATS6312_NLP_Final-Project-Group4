# Code for NLP Political Bias Detection Project

**Data:**

* This folder contains .JSON data records used for model training. 

**Model Scripts:**

* **RoBERTA_model.py:** 
    * Implements the RoBERTA transformer model for political bias detection.
    * Before running, ensure you have the 'transformers' library installed (`pip install transformers`).
    * You may need to adjust file paths within the script if you're working locally.
* **Classical_Models.py:**
    * Contains implementations of classical machine learning models (Naive Bayes and Logistic Regression).

**Code Execution:**

* The Python scripts (.py files) in this folder are divided into cells using the `#%%` marker. This allows for flexible execution:
    * **Running the whole file:** Execute the entire script for complete model training or prediction processes (Before execution make sure that path wich you load data and save model are the ones that desired).
    * **Running individual cells:** Focus on specific code blocks for experimentation, testing, or debugging.

**Recommended Usage:**

We recommend running cells individually during development and experimentation. This enables you to isolate model components, make changes, and observe their effects in a granular manner or keep a clean track of command executions.


**Saved Model:**

* **model.pkl:**  Represents a pre-trained RoBERTA model.  This file can be loaded for prediction tasks or to continue fine-tuning.


