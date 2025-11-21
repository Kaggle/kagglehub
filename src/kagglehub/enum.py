"""Functions to convert string to kagglesdk enum values."""

import enum
import re
from typing import TypeVar

T = TypeVar("T")


# TODO(b/461859420): Consider moving to kagglesdk
def to_enum(enum_class: type[T], enum_str: str) -> T:
    enum_key = _camel_to_snake(enum_str).upper()

    try:
        return getattr(enum_class, enum_key)
    except AttributeError:
        try:
            prefix = _camel_to_snake(enum_class.__name__).upper()
            full_name = f"{prefix}_{enum_key}"
            return getattr(enum_class, full_name)
        except AttributeError:
            # Handle PY_TORCH vs PYTORCH, etc.
            full_name = full_name.replace("_", "")
            for item in vars(enum_class):
                if item.replace("_", "") == full_name:
                    return getattr(enum_class, item)
            msg = f"'{enum_str}' is not a valid ModelFramework"
            raise ValueError(msg) from None


def enum_to_str(enum: enum.Enum) -> str:
    names = str(enum).split(".")
    enum_class_name = names[0]
    enum_value_name = names[1]
    snakecase_prefix = _camel_to_snake(enum_class_name).upper()
    enum_value_name = enum_value_name.removeprefix(snakecase_prefix + "_")
    return _snake_to_camel(enum_value_name)


def _camel_to_snake(value: str) -> str:
    value = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def _snake_to_camel(value: str) -> str:
    camel_str = "".join(word.capitalize() for word in value.lower().split("_"))
    # we lower only the first character
    return value[0].lower() + camel_str[1:]
