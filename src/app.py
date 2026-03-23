import os
import gradio as gr
from src.chain import create_chain, ask
from config import APP_TITLE, APP_PORT

# Bypass system proxy for localhost so Gradio can reach itself
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Initialize chain and chat history once at startup
chain = create_chain()
chat_history = []


def chat(question, history):
    """Handle a single chat turn."""
    global chat_history
    answer, chat_history = ask(chain, question, chat_history)
    return answer


def main():
    """Launch the Gradio chat interface."""
    demo = gr.ChatInterface(
        fn=chat,
        title=APP_TITLE,
        description="Internal assistant for Dragon Palace recipes, allergens, and kitchen operations. Ask about ingredients, allergen risks, or preparation methods.",
        type="messages",
        examples=[
            "What allergens are in the Golden Dragon Dumplings?",
            "Does the Midnight Black Garlic Noodles contain nuts?",
            "What's the secret ingredient in Phoenix Fire Wings glaze?",
            "Which dishes share fryer cross-contamination risk?",
            "Is the Jade Emperor's Soup safe for vegetarians?",
            "What hidden allergens should servers know about?",
        ],
    )
    demo.launch(
        server_port=APP_PORT,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
