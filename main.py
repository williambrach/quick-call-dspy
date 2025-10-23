import os

import dspy
from dotenv import load_dotenv

import signaturize

load_dotenv()

# load lm for dspy
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

lm = dspy.LM("azure/gpt-4.1-mini-sweden", api_key=API_KEY, api_base=API_BASE)
dspy.configure(lm=lm)


def get_input_fields(signature: type[dspy.Signature]) -> list[str]:
    input_fields = []
    for field_name in signature.fields:
        field = signature.fields[field_name]
        if field.json_schema_extra.get("__dspy_field_type", None) == "input":  # type: ignore
            input_fields.append(field_name)
    return input_fields


def main() -> None:
    class_string = """
class query_signature(dspy.Signature):
    query: str = dspy.InputField(
        desc="query from the user"
    )
    answer: str = dspy.OutputField(
        desc="answer to the user"
    )
    """
    try:
        signature = signaturize.from_dspy_string(class_string)
        program = dspy.Predict(signature)
        print(get_input_fields(signature))
        print(program(query="What is the capital of France?"))

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
