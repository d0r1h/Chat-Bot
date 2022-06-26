from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import torch


app = Flask(__name__)



tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    
    def predict(input, history=[]):
      new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
      bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)
      history = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id).tolist()
      response = tokenizer.decode(history[0]).replace("<|endoftext|>", "\n")
      return response

    return predict(msg)


if __name__ == "__main__":
    app.run()

