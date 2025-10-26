# this file is provided from- https://raw.githubusercontent.com/Archelunch/vibe-dspy/refs/heads/main/src/signature_generator.py
# please check for updates there

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import dspy
from pydantic import BaseModel, Field, ValidationError


class FieldType(str, Enum):
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    LIST_STRING = "list[str]"
    LIST_INT = "list[int]"
    LIST_FLOAT = "list[float]"
    DICT_STR_STR = "dict[str, str]"
    DICT_STR_INT = "dict[str, int]"
    DICT_STR_ANY = "dict[str, Any]"
    IMAGE = "dspy.Image"
    AUDIO = "dspy.Audio"
    LITERAL = "Literal"
    OPTIONAL_STR = "Optional[str]"
    OPTIONAL_INT = "Optional[int]"
    OPTIONAL_FLOAT = "Optional[float]"


class FieldRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class GeneratedField(BaseModel):
    name: str = Field(description="The field name (snake_case, descriptive)")
    type: FieldType = Field(description="The Python type for this field")
    role: FieldRole = Field(description="Whether this is an input or output field")
    description: str = Field(description="Description of what this field represents")
    literal_values: Optional[List[str]] = Field(
        default=None, description="For Literal types, the allowed values"
    )
    default_value: Optional[str] = Field(
        default=None,
        description="Default value for the field (as string representation)",
    )

    def to_dspy_field_code(self) -> str:
        if self.type == FieldType.LITERAL and self.literal_values:
            type_annotation = (
                f"Literal[{', '.join(repr(v) for v in self.literal_values)}]"
            )
        else:
            type_annotation = self.type.value

        field_type = "InputField" if self.role == FieldRole.INPUT else "OutputField"

        if self.description:
            return f'{self.name}: {type_annotation} = dspy.{field_type}(desc="{self.description}")'
        else:
            return f"{self.name}: {type_annotation} = dspy.{field_type}()"


class SignatureGeneration(dspy.Signature):
    prompt: str = dspy.InputField(
        desc="Natural language description of the desired functionality"
    )
    task_description: str = dspy.OutputField(
        desc="Clear description of what the signature accomplishes"
    )
    signature_fields: list[GeneratedField] = dspy.OutputField(
        desc="List of input and output fields for the signature"
    )
    signature_name: str = dspy.OutputField(
        desc="Suggested class name for the signature (PascalCase)"
    )


class SignatureGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(SignatureGeneration)

    def forward(self, prompt: str):
        """Generate DSPy signature and return raw prediction attributes"""
        result = self.generator(prompt=prompt)

        return dspy.Prediction(
            signature_name=result.signature_name,
            task_description=result.task_description,
            signature_fields=result.signature_fields,
            reasoning=result.reasoning if hasattr(result, "reasoning") else None,
        )

    def generate_signature(self, prompt: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility - returns formatted dict"""
        try:
            result = self.forward(prompt=prompt)

            return {
                "signature_name": result.signature_name,
                "task_description": result.task_description,
                "fields": [field.model_dump() for field in result.signature_fields],
                "code": self.generate_code(result),
                "reasoning": result.reasoning,
            }
        except ValidationError as e:
            error_msg = f"Data validation error from Pydantic: {e}"
            return self._format_error(error_msg)
        except Exception as e:
            # Catch other potential errors (e.g., from dspy)
            error_msg = f"An unexpected error occurred: {e}"
            return self._format_error(error_msg)

    def _format_error(self, error_message: str) -> Dict[str, Any]:
        """Helper to create a standardized error dictionary."""
        return {
            "error": error_message,
            "signature_name": None,
            "task_description": None,
            "fields": [],
            "code": None,
        }

    @classmethod
    def create_signature_class(cls, prediction: dspy.Prediction) -> type:
        """
        Dynamically creates a dspy.Signature class from a prediction object.

        Args:
            prediction: An object with attributes `signature_name`, `task_description`,
                        and `signature_fields`.

        Returns:
            A new class that inherits from dspy.Signature.
        """
        class_name = prediction.signature_name
        docstring = prediction.task_description

        class_attrs = {"__doc__": docstring, "__annotations__": {}}

        for field in prediction.signature_fields:
            field_name = field.name
            py_type = cls._get_python_type_from_field(field)

            dspy_field_class = (
                dspy.InputField if field.role == FieldRole.INPUT else dspy.OutputField
            )
            dspy_field_instance = dspy_field_class(desc=field.description)

            class_attrs[field_name] = dspy_field_instance
            class_attrs["__annotations__"][field_name] = py_type

        DynamicSignature = type(class_name, (dspy.Signature,), class_attrs)
        return DynamicSignature

    @staticmethod
    def _get_python_type_from_field(field: "GeneratedField") -> type:
        """Converts a GeneratedField into a Python type for annotations."""
        type_str = field.type.value

        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list[str]": List[str],
            "list[int]": List[int],
            "list[float]": List[float],
            "dict[str, str]": Dict[str, str],
            "dict[str, int]": Dict[str, int],
            "dict[str, Any]": Dict[str, Any],
            "Optional[str]": Optional[str],
            "Optional[int]": Optional[int],
            "Optional[float]": Optional[float],
            "dspy.Image": dspy.Image,
            "dspy.Audio": dspy.Audio,
        }

        if field.type == FieldType.LITERAL and field.literal_values:
            return Literal[tuple(field.literal_values)]

        if type_str in type_map:
            return type_map[type_str]

        raise TypeError(
            f"Unsupported field type for dynamic class creation: {type_str}"
        )

    @classmethod
    def generate_code(cls, prediction) -> str:
        """Generate Python code from a signature prediction"""
        imports = cls.get_required_imports(prediction.signature_fields)

        code_lines = []
        # code_lines.extend(imports)
        # code_lines.append("")
        code_lines.append(f"class {prediction.signature_name}(dspy.Signature):")
        code_lines.append(f'    """{prediction.task_description}"""')
        code_lines.append("")

        for field in prediction.signature_fields:
            code_lines.append(f"    {field.to_dspy_field_code()}")

        return "\n".join(code_lines)

    @classmethod
    def get_required_imports(cls, fields: List[GeneratedField]) -> List[str]:
        """Determine required imports based on field types"""
        imports = ["import dspy"]
        typing_imports = set()

        for field in fields:
            if field.type == FieldType.LITERAL:
                typing_imports.add("Literal")
            elif field.type in [
                FieldType.OPTIONAL_STR,
                FieldType.OPTIONAL_INT,
                FieldType.OPTIONAL_FLOAT,
            ]:
                typing_imports.add("Optional")
            elif field.type in [
                FieldType.LIST_STRING,
                FieldType.LIST_INT,
                FieldType.LIST_FLOAT,
            ]:
                typing_imports.add("List")
            elif field.type in [
                FieldType.DICT_STR_STR,
                FieldType.DICT_STR_INT,
                FieldType.DICT_STR_ANY,
            ]:
                typing_imports.add("Dict")

            if field.type == FieldType.DICT_STR_ANY:
                typing_imports.add("Any")

        if typing_imports:
            imports.append(
                f"from typing import {', '.join(sorted(list(typing_imports)))}"
            )

        return imports