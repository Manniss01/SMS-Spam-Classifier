import gradio as gr
from src.predict import predict_spam


def predict_with_label(message: str) -> str:
    message = message.strip()
    if not message:
        return "⚠️ Please enter a message."

    prediction = predict_spam(message)
    return "🛑 Spam" if prediction == "spam" else "✅ Not Spam"


examples = [
    ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575."],
    ["Hey, are we still meeting tomorrow?"],
    ["Thanks for your subscription to Ringtone UK. Your mobile will be charged £5/month. Reply YES to confirm or NO to cancel."],
]

custom_css = """
body {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: auto;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    padding: 2rem;
}
textarea {
    border: 2px solid #007bff !important;
    border-radius: 10px !important;
    padding: 10px !important;
    font-size: 16px !important;
}
.output-textbox {
    font-size: 20px;
    font-weight: bold;
    color: #333;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #888;
    margin-top: 30px;
}
"""

iface = gr.Interface(
    fn=predict_with_label,
    inputs=gr.Textbox(label="📩 SMS Message", lines=5, placeholder="Type your SMS message here..."),
    outputs=gr.Text(label="🔎 Prediction"),
    title="📨 SMS Spam Classifier",
    description=(
        "🔍 A simple yet effective tool that uses a machine learning model to classify SMS messages "
        "as **Spam** or **Not Spam**. Just enter a message to get instant results!\n\n"
        "<div class='footer'> Developed by <strong>Unicodax</strong></div>"
    ),
    examples=examples,
    allow_flagging="never",
    cache_examples=True,
    css=custom_css
)

if __name__ == "__main__":
    iface.launch()
