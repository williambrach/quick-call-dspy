import json
import os

import dspy
import gradio as gr
from dotenv import load_dotenv
from dspy.signatures.signature import Signature

import signaturize

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME")


def get_fields_by_type(signature: type[Signature], field_type: str) -> list[str]:
    """Extracts field names from a dspy.Signature based on their type."""
    return [
        name
        for name, field in signature.fields.items()
        if field.json_schema_extra.get("__dspy_field_type") == field_type  # type: ignore
    ]


EXAMPLE_CLASS_STRING = """
class BasicQA(dspy.Signature):
    \"\"\"Answer questions with short factoid answers.\"\"\"
    question: str = dspy.InputField(desc="The question to answer.")
    answer: str = dspy.OutputField(desc="A concise answer to the question.")
"""

def configure_lm(
    api_key: str | None, api_base: str | None, model_name: str | None
) -> tuple[str, dspy.LM | None]:
    """Configures the dspy.LM with user-provided credentials."""

    if model_name is None or api_key is None or api_base is None:
        return "❌ Error: Model Name, API Key, and API Base are required.", None
    try:
        lm = dspy.LM(model_name, api_key=api_key, api_base=api_base)
        return "✅ LM Configured Successfully!", lm
    except Exception as e:
        print(f"Configuration error: {e}")
        return f"❌ Configuration Error: {e}", None


def generate_signature(
    signature_string: str, prompt: str, mode: str, lm: dict | None
) -> tuple[str, dspy.Predict | None, str, str]:
    if not lm:
        return (
            "",
            None,
            json.dumps({}, indent=2),
            "❌ Error: Configure the LM in the 'Configuration' tab first.",
        )
    try:
        if mode == "prompt":

            lm = lm.get("configured")
            with dspy.context(lm=lm):
                signature_string = str(
                    signaturize.from_prompt(prompt, return_type="string")
                )
        signature = signaturize.from_dspy_string(signature_string)
        program = dspy.Predict(signature)

        input_fields = get_fields_by_type(signature, "input")
        input_template = dict.fromkeys(input_fields, "...")
        input_template = json.dumps(input_template, indent=2)

        status = "✅ Signature Loaded. Ready to run."
        return signature_string, program, input_template, status

    except Exception as e:
        return signature_string, None, json.dumps({}, indent=2), f"❌ Error: {e}"


def run_prediction(
    program: dspy.Predict | None, input_data: str, lm: dict | None
) -> tuple[dict, str]:

    if program is None:
        return {}, "❌ Error: No signature is loaded. Please generate one first."

    if lm is None or lm.get("configured") is None:
        return {}, "❌ Error: LM is not configured. Please configure it first."

    lm = lm.get("configured")
    input_data = json.loads(input_data)

    if not isinstance(input_data, dict):
        return {}, "❌ Error: Input must be valid JSON."

    try:
        # Run the prediction
        with dspy.context(lm=lm):
            result = program(**input_data)

            # Extract output fields into a dictionary
            output_fields = get_fields_by_type(program.signature, "output")
            output_dict = {field: getattr(result, field) for field in output_fields}

            return output_dict, "✅ Prediction Complete."

    except Exception as e:
        return {}, f"❌ Error during prediction: {e}"


# --- Gradio App ---

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# quick-call-dspy")

    dspy_program = gr.State(None)
    _, state = configure_lm(API_KEY, API_BASE, MODEL_NAME)
    lm_configured_state = gr.State({"configured": state})

    run_status_md = gr.Markdown()
    # --- LM Configuration ---
    with gr.Accordion("LM Configuration", open=False):
        gr.Markdown(
            "Provide your LM credentials. These are used for generating signatures from prompts and for running predictions."
        )
        api_key_box = gr.Textbox(
            label="API Key", type="password", value=os.getenv("API_KEY", None)
        )
        api_base_box = gr.Textbox(
            label="API Base", type="password", value=os.getenv("API_BASE", None)
        )
        model_name_box = gr.Textbox(
            label="Model Name",
            value=os.getenv("MODEL_NAME", None),
        )
        config_button = gr.Button("Configure LM")
        config_status_md = gr.Markdown()

    with gr.Accordion("Suggest signature", open=False):
        gr.Markdown("Enter prompt to generate Signature")
        with gr.Row():
            with gr.Column():
                prompt_box = gr.Textbox(
                    label="Prompt to generate signature",
                    lines=5,
                    placeholder="e.g., Create a signature for generating recipes. Output fields should be ingredients, steps, and title.",
                )
            with gr.Column():
                gen_prompt_button = gr.Button(
                    "Generate from Prompt",
                    variant="primary",
                    size="lg",
                )

    gr.Markdown("# DSPy Signature")
    with gr.Row():
        with gr.Column():
            string_box = gr.Code(
                label="dspy.Signature Class",
                language="python",
                lines=10,
                value=EXAMPLE_CLASS_STRING.strip(),
            )
        with gr.Column():
            gen_string_button = gr.Button("Generate from String", variant="primary")

    # # --- Prediction ---
    gr.Markdown("# Prediction")
    with gr.Row():
        with gr.Column():
            input_json = gr.Code(label="Inputs", language="json", interactive=True)
            run_button = gr.Button("Run Prediction", variant="primary")
        with gr.Column():
            output_json = gr.JSON(label="Output")

    # --- Event Listeners ---

    # Configuration
    config_button.click(
        fn=configure_lm,
        inputs=[api_key_box, api_base_box, model_name_box],
        outputs=[config_status_md, lm_configured_state],
    )

    # Signature Generation
    gen_string_button.click(
        fn=lambda string_box, prompt_box, lm_state: generate_signature(
            string_box, prompt_box, "string", lm_state
        ),
        inputs=[string_box, prompt_box, lm_configured_state],
        outputs=[string_box, dspy_program, input_json, run_status_md],
    )

    gen_prompt_button.click(
        fn=lambda string_box, prompt_box, lm_state: generate_signature(
            string_box, prompt_box, "prompt", lm_state
        ),
        inputs=[string_box, prompt_box, lm_configured_state],
        outputs=[string_box, dspy_program, input_json, run_status_md],
    )

    # Prediction
    run_button.click(
        fn=run_prediction,
        inputs=[dspy_program, input_json, lm_configured_state],
        outputs=[output_json, run_status_md],
    )

if __name__ == "__main__":
    demo.launch()
