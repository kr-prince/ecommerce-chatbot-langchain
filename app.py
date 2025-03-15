import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

from utils.agent_utils import create_primary_assistant_runnable_and_build_graph
from utils.configs import ENV_FILE_PATH

SYSTEM_INTERRUPT_FLAG = False
def sync_chatbot_response(message, history, graph, config):
    """Synchronous chatbot response generator."""
    global SYSTEM_INTERRUPT_FLAG
    try:
        if not SYSTEM_INTERRUPT_FLAG:
            response = graph.invoke({
                "messages": [("user", message)]
            }, config)['messages'][-1].content

            bot_response = response
            snapshot = graph.get_state(config)
            if snapshot.next:
                # This is for interruption
                print(f"Coming here: {snapshot.next}")
                SYSTEM_INTERRUPT_FLAG = True
                response = "Should I generate the RA number for you? yes/no"
                bot_response = response
            history.append((f"👤 {message}", f"🤖 {bot_response}"))
        else:
            if message.lower().startswith("yes"):
                response = graph.invoke(
                    None,
                    config,
                )['messages'][-1].content
            else:
                snapshot = graph.get_state(config)
                response = graph.invoke({
                    "messages": [
                        ToolMessage(
                            # tool_call_id=response['messages'][-1].tool_call_id,
                            tool_call_id=snapshot.values['messages'][-1].tool_calls[0]['id'],
                            content=f"Generate-Return-Authorization denied by user. Reasoning: '{message}'. Proceed with last conversation.",
                        )]
                },
                config,
            )['messages'][-1].content
            bot_response = response
            history.append((f"👤 {message}", f"🤖 {bot_response}"))
            SYSTEM_INTERRUPT_FLAG = False
        return history, ""
    except Exception as e:
        print(f"Sync Error: {e}")
        history.append((f"👤 {message}", f"⚠️ Error: {str(e)}"))
        return history, ""

if __name__ == "__main__":
    load_dotenv(ENV_FILE_PATH)
    graph = create_primary_assistant_runnable_and_build_graph()
    config = {
        "configurable": {
            "thread_id": 1,
        },
    }
    
    # Gradio UI with a fixed chat window width
    with gr.Blocks(
        theme=gr.themes.Soft(),css="""
        body {background-color: #f8f9fa; font-family: 'Arial', sans-serif;}
        .gradio-container {max-width: 700px; margin: auto; text-align: center;}  /* Centering everything */
        #chatbot {border-radius: 10px; background: white; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                width: 100%; min-height: 450px; max-width: 700px; margin: auto;} /* Fixed width from start */
        .message-box {width: 100%;} /* Fix input width */
    """) as demo:

        gr.Markdown("## 👟 **SoleMate – Your Footwear Assistant**", elem_id="banner")

        chatbot = gr.Chatbot(elem_id="chatbot", height=450, type="tuples")

        msg = gr.Textbox(placeholder="👋 Ask me about returns, exchanges, or orders...", label="", elem_classes="message-box")

        send_btn = gr.Button("⌯⌲ Send", variant="primary")

        # Sync response handler with message clearing
        send_btn.click(
            fn=lambda message, history: sync_chatbot_response(message, history, graph, config),
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            queue=True
        )
        msg.submit(
            fn=lambda message, history: sync_chatbot_response(message, history, graph, config),
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            queue=True
        )

    # Launch the UI
    demo.launch(share=True)
