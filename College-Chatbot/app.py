from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the SVM model and label mapping
with open('label_mapping_SVM.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

with open('qa_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    
    # Preprocess the query if necessary (e.g., vectorize)
    # For example, if you need to vectorize the query using TF-IDF:
    # vectorizer = ... (load or define your vectorizer)
    # query_vector = vectorizer.transform([query])
    
    query_vector = np.array([query])  # Assuming the model can take raw text
    
    prediction = model.predict(query_vector)[0]
    response = label_mapping.get(prediction, "Unknown response")
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
