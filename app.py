from flask import Flask, request, jsonify
import pickle
from data_loader import clean_input_sentence
# Make sure this is the *correct* `predict` function (the one you provided)
from prediction_function import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
def load_model():
    with open("naive_bayes_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# Initialize model and extract components
model_dict = load_model()
logprior = model_dict["logprior"]
loglikelihood = model_dict["loglikelihood"]
vocab = model_dict["vocab"]

# Target classes MUST match what the model was trained with, and what the `predict` function returns
target_classes = [-1, 0, 1]  # -1, 0, and 1 if your model produces -1, 0, and 1
# Label mapping
label_mapping = {
    0: "Neutral",    # 0: Neutral
    -1: "Hate Speech",  # -1: Hate Speech
    1: "Not Hate Speech" # 1: Not Hate Speech
}


@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        # Get the input data
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400

        # Get and clean the input text
        input_text = data['text']
        cleaned_input = clean_input_sentence(input_text)

        # Make prediction
        prediction = predict(
            [" ".join(cleaned_input)],
            logprior,
            loglikelihood,
            target_classes,
            vocab
        )[0]  # Get first prediction since we're only sending one text

        # Prepare response
        response = {
            'text': input_text,
            'prediction': label_mapping[prediction],
            'prediction_code': int(prediction),
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': all(x is not None for x in [logprior, loglikelihood, vocab])
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)