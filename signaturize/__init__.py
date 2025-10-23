import dspy


def from_dspy_string(cls_string: str) -> type[dspy.Signature]:
    try:
        namespace = {"dspy": dspy}
        exec(cls_string, namespace)
        signature_classes = []
        for obj in namespace.values():
            if isinstance(obj, dspy.SignatureMeta) and obj is not dspy.Signature:
                signature_classes.append(obj)

        # We must find exactly one class.
        if not signature_classes:
            raise ValueError("No class inheriting from dspy.Signature was found in the provided string.")

        if len(signature_classes) > 1:
            class_names = [cls.__name__ for cls in signature_classes]
            raise ValueError(
                f"Multiple dspy.Signature classes found: {', '.join(class_names)}. "
                "The string must define exactly one."
            )
        return signature_classes[0]

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e

