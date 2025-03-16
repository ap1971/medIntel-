import React, { useState } from "react";
import axios from "axios";

const App = () => {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResponse(res.data);
    } catch (error) {
      console.error("Upload error:", error);
      alert("Failed to process the file.");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        <h1 className="text-xl font-bold mb-4">Upload Your Form</h1>
        <input
          type="file"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-900 border rounded-lg cursor-pointer mb-4"
        />
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 w-full"
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload"}
        </button>
      </div>

      {response && (
        <div className="bg-white p-6 rounded-lg shadow-md mt-6 w-full max-w-2xl">
          <h2 className="text-lg font-semibold">Extracted Text</h2>
          <p className="text-sm bg-gray-100 p-2 rounded">{response.text}</p>

          {response.errors.length > 0 ? (
            <div className="mt-4">
              <h2 className="text-lg font-semibold">Errors & Suggestions</h2>
              <ul className="list-disc pl-5 text-red-500">
                {response.errors.map((error, index) => (
                  <li key={index}>
                    <strong>Error:</strong> {error} <br />
                    <strong>Suggestion:</strong> {response.suggestions[index]}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="mt-4 text-green-600">No errors detected! 🎉</p>
          )}
        </div>
      )}
    </div>
  );
};

export default App;
