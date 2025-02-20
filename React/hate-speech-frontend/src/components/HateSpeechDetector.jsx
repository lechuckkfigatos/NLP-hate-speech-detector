import React, { useState } from "react";

const HateSpeechDetector = () => {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = () => {
    setLoading(true);
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: inputText }),
    })
      .then((response) => response.json())
      .then((data) => {
        setTimeout(() => {
          setResult(data);
          setLoading(false);
        }, 2000);
      })
      .catch((error) => {
        console.error("Error:", error);
        setLoading(false);
      });
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !loading) {
      event.preventDefault(); // Prevent newline in textarea
      handleAnalyze();
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-900 text-white">
      <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl max-w-md w-full">
        <h1 className="text-3xl font-bold mb-6 text-center">
          Hate Speech Detector
        </h1>
        <div className="mb-4">
          <textarea
            className="w-full p-4 rounded-lg bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Enter text to analyze..."
            style={{ width: "549px", height: "213px" }}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown} // Add the keydown handler
          />
        </div>
        <button
          onClick={handleAnalyze}
          className={`w-full py-2 rounded-lg transition-all font-semibold ${
            loading
              ? "bg-indigo-400 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-700"
          }`}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Text"}
        </button>
        {result && (
          <div className="mt-6 text-center">
            <h2
              className={`text-xl font-semibold ${
                result.prediction_code === 1
                  ? "text-red-500"
                  : "text-green-500"
              }`}
            >
              {result.prediction}
            </h2>
            <p className="text-sm mt-2 text-gray-400">
              This text was classified as:{" "}
              <span
                className={`font-bold ${
                  result.prediction_code === 1
                    ? "text-red-400"
                    : "text-green-400"
                }`}
              >
                {result.prediction}
              </span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HateSpeechDetector;