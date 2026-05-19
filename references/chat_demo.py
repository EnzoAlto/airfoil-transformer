import gradio as gr


def respond(message, history):
    lower = message.lower()

    if "hello" in lower or "hi" in lower:
        return "Hello! What are we building today?"
    if "airfoil" in lower:
        return "Airfoils, nice. Are you analyzing geometry, pressure, or lift?"
    if "bye" in lower:
        return "Bye! See you next run."

    return f"I received: {message}"


gr.ChatInterface(
    fn=respond,
    title="Airfoil Assistant",
    # type="messages",
).launch()