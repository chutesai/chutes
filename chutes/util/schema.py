import inspect
from typing import (
    get_type_hints,
    Dict,
    Any,
    List,
    Set,
    Type,
    Tuple,
    Union,
    Literal,
    get_args,
    get_origin,
)
from pydantic import BaseModel


class SchemaExtractor:
    """
    Helper class to handle schema extraction logic.
    """

    PYTHON_TYPE_TO_JSON_TYPE = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array", "items": {}},
        dict: {"type": "object"},
        None: {"type": "null"},
    }

    @classmethod
    def get_minimal_schema(cls, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Minimal schema helper without function introspection.
        """
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            raise ValueError("minimal_input_schema must be a Pydantic model")
        model_schema = model.model_json_schema(ref_template="#/definitions/{model}")
        schema = {
            "type": "object",
            "properties": {"input_args": {"$ref": f"#/definitions/{model.__name__}"}},
            "required": ["input_args"],
            "definitions": {
                model.__name__: {
                    k: v for k, v in model_schema.items() if k not in ("title", "definitions")
                }
            },
        }
        if "definitions" in model_schema:
            schema["definitions"].update(model_schema["definitions"])
        return schema

    @classmethod
    def _get_schema_for_type(cls, type_hint: Any, definitions: Dict) -> Dict[str, Any]:
        """
        Convert Python types and typing annotations to OpenAPI JSON schemas.
        """
        if type_hint is None:
            return {"type": "null"}
        if inspect.isclass(type_hint) and issubclass(type_hint, BaseModel):
            model_schema = type_hint.model_json_schema(ref_template="#/definitions/{model}")
            model_name = type_hint.__name__
            if model_name not in definitions:
                definitions[model_name] = {
                    k: v for k, v in model_schema.items() if k not in ("title", "definitions")
                }
                if "definitions" in model_schema:
                    definitions.update(model_schema["definitions"])
            return {"$ref": f"#/definitions/{model_name}"}
        if type_hint in cls.PYTHON_TYPE_TO_JSON_TYPE:
            return cls.PYTHON_TYPE_TO_JSON_TYPE[type_hint]

        origin = get_origin(type_hint)
        args = get_args(type_hint)
        if origin is Union:
            schemas = [cls._get_schema_for_type(arg, definitions) for arg in args]
            if len(schemas) == 2 and {"type": "null"} in schemas:
                non_null_schema = next(s for s in schemas if s != {"type": "null"})
                return {**non_null_schema, "nullable": True}
            return {"oneOf": schemas}

        elif origin is list or origin is List:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": cls._get_schema_for_type(item_type, definitions),
            }

        elif origin is dict or origin is Dict:
            if not args:
                return {"type": "object"}
            key_type, value_type = args
            if key_type is not str:
                key_type = str
            return {
                "type": "object",
                "additionalProperties": cls._get_schema_for_type(value_type, definitions),
            }

        elif origin is tuple or origin is Tuple:
            return {
                "type": "array",
                "items": [cls._get_schema_for_type(arg, definitions) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }

        elif origin is set or origin is Set:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": cls._get_schema_for_type(item_type, definitions),
                "uniqueItems": True,
            }

        elif origin is Literal:
            return {"enum": list(args)}

        return {}

    @classmethod
    def extract_schemas(cls, func) -> Dict[str, Any]:
        """
        Extract input and output schemas from function signature.
        """
        hints = get_type_hints(func)
        return_type = hints.pop("return", None)
        input_fields = {}
        definitions = {}
        for param_name, param_type in hints.items():
            input_fields[param_name] = cls._get_schema_for_type(param_type, definitions)
        input_schema = {
            "type": "object",
            "properties": input_fields,
            "required": list(input_fields.keys()),
            "definitions": definitions,
        }
        output_schema = None
        if return_type:
            output_schema = cls._get_schema_for_type(return_type, definitions)
            if "$ref" in output_schema:
                output_schema = {
                    **{"type": "object"},
                    **definitions[output_schema["$ref"].split("/")[-1]],
                }
        return input_schema, output_schema

    @classmethod
    def extract_models(cls, func) -> List[Type[BaseModel]]:
        """
        Extract pydantic models from function signature.
        """
        models = []
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        params = list(sig.parameters.items())
        if params and (params[0][0] == "self" or params[0][0] not in hints):
            params = params[1:]
        for param_name, _ in params:
            if param_name not in hints:
                return None
            type_hint = hints[param_name]
            if not (inspect.isclass(type_hint) and issubclass(type_hint, BaseModel)):
                return None
            models.append(type_hint)
        return models
