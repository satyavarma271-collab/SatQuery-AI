import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(
    "Upload an image and ask a question to begin analysis."
  );
  const [loading, setLoading] = useState(false);

  const handleImage = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);

      const imageUrl = URL.createObjectURL(selectedFile);
      setImage(imageUrl);

      setAnswer(
        "Image uploaded successfully. Now enter your question and click Analyze Image."
      );
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setAnswer("Please upload a satellite image first.");
      return;
    }

    if (!question.trim()) {
      setAnswer("Please enter a question about the image.");
      return;
    }

    setLoading(true);
    setAnswer("🤖 AI is analyzing the image...");

    const formData = new FormData();

    formData.append("image", file);
    formData.append("question", question);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      setAnswer(data.answer);
    } catch (error) {
      console.error(error);

      setAnswer(
        "❌ Could not connect to the AI backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      <header className="navbar">
        <div className="logo">
          🛰️ SatQuery AI
        </div>

        <div className="nav-text">
          Remote Sensing Intelligence
        </div>
      </header>


      <main className="container">

        <section className="hero">
          <h1>Ask Questions About Satellite Images</h1>

          <p>
            Upload a remote sensing image and interact with it
            using natural language.
          </p>
        </section>


        <section className="workspace">

          <div className="upload-card">

            <h2>🖼️ Satellite Image</h2>

            <label className="upload-box">

              {image ? (
                <img
                  src={image}
                  alt="Uploaded satellite preview"
                />
              ) : (
                <>
                  <div className="upload-icon">
                    📤
                  </div>

                  <h3>
                    Upload Satellite Image
                  </h3>

                  <p>
                    JPG, PNG or remote sensing image
                  </p>
                </>
              )}

              <input
                type="file"
                accept="image/*"
                onChange={handleImage}
                hidden
              />

            </label>

          </div>


          <div className="query-card">

            <h2>💬 Ask SatQuery AI</h2>

            <textarea
              placeholder="Example: What objects are visible in this image?"
              value={question}
              onChange={(e) =>
                setQuestion(e.target.value)
              }
            />

            <button
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading
                ? "🔄 Analyzing..."
                : "🔍 Analyze Image"}
            </button>

          </div>

        </section>


        <section className="answer-card">

          <h2>🤖 AI Analysis</h2>

          <div className="answer-box">

            <p>{answer}</p>

          </div>

        </section>

      </main>

    </div>
  );
}

export default App;