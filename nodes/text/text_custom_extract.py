import ast
import json
import re
import traceback

from comfy_api.latest import io


class TextCustomExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_TextCustomExtract",
            display_name="Text Custom Extract",
            category="1hewNodes/text",
            inputs=[
                io.String.Input("json_data", default="", multiline=True),
                io.String.Input("key", default="zh"),
                io.Combo.Input("precision_match", options=["disabled", "enabled"], default="disabled"),
                io.String.Input("label_filter", default=""),
            ],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    async def execute(
        cls, json_data: str, key: str, precision_match: str, label_filter: str
    ) -> io.NodeOutput:
        try:
            data = cls._parse_json_data(json_data)
            filters = cls._parse_label_filters(label_filter)

            values: list[str] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    if filters:
                        label_value = cls._find_key_in_object(item, "label", "disabled")
                        if not cls._matches_label_filter(label_value, filters):
                            continue
                    value = cls._find_key_in_object(item, key, precision_match)
                    if value is not None:
                        values.append(cls._format_value(value))
                result = "\n".join(values) if values else ""
            elif isinstance(data, dict):
                if filters:
                    label_value = cls._find_key_in_object(data, "label", "disabled")
                    if not cls._matches_label_filter(label_value, filters):
                        result = ""
                    else:
                        value = cls._find_key_in_object(data, key, precision_match)
                        result = (
                            cls._format_value(value) if value is not None else ""
                        )
                else:
                    value = cls._find_key_in_object(data, key, precision_match)
                    result = cls._format_value(value) if value is not None else ""
            else:
                result = ""

            return io.NodeOutput(result)
        except Exception:
            traceback.print_exc()
            return io.NodeOutput("")

    @classmethod
    def _clean_input_text(cls, text: str) -> str:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]

        start_pos = -1
        for i, char in enumerate(text):
            if char in "[{":
                start_pos = i
                break
        if start_pos != -1:
            text = text[start_pos:]

        end_pos = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in "]}":
                end_pos = i + 1
                break
        if end_pos != -1:
            text = text[:end_pos]
        return text

    @classmethod
    def _parse_json_data(cls, json_data):
        text = (
            cls._clean_input_text(json_data)
            if isinstance(json_data, str)
            else str(json_data)
        )
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            parsed = cls._try_literal_eval(text)
            if parsed is not None:
                return parsed

            parsed = cls._try_json_loads(text.replace("'", '"'))
            if parsed is not None:
                return parsed

            parsed = cls._try_loose_parse(text)
            if parsed is not None:
                return parsed

            pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, text)
            if matches:
                return {key: value for key, value in matches}
            return {}

    @classmethod
    def _try_json_loads(cls, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @classmethod
    def _try_literal_eval(cls, text: str):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    @classmethod
    def _try_loose_parse(cls, text: str):
        stripped = text.strip()
        if not stripped:
            return None

        start_idx = None
        start_char = None
        for idx, ch in enumerate(stripped):
            if ch in "[{":
                start_idx = idx
                start_char = ch
                break
        if start_idx is None or start_char is None:
            return None

        end_char = "}" if start_char == "{" else "]"
        end_idx = stripped.rfind(end_char)
        if end_idx == -1 or end_idx <= start_idx:
            return None

        container = stripped[start_idx : end_idx + 1]
        try:
            if start_char == "{":
                parsed, _ = cls._loose_parse_dict(container, 0)
                return parsed
            parsed, _ = cls._loose_parse_list(container, 0)
            return parsed
        except Exception:
            return None

    @classmethod
    def _loose_parse_dict(cls, text: str, start: int):
        if start >= len(text) or text[start] != "{":
            raise ValueError("Expected '{'")

        idx = start + 1
        result = {}
        while idx < len(text):
            idx = cls._skip_ws(text, idx)
            if idx >= len(text):
                break
            if text[idx] == "}":
                return result, idx + 1
            if text[idx] == ",":
                idx += 1
                continue

            key, idx = cls._read_key(text, idx)
            idx = cls._skip_ws(text, idx)
            if idx < len(text) and text[idx] == ":":
                idx += 1
            idx = cls._skip_ws(text, idx)

            value, idx = cls._loose_parse_value(text, idx)
            if key:
                result[key] = value

            idx = cls._skip_ws(text, idx)
            if idx < len(text) and text[idx] == ",":
                idx += 1
        return result, idx

    @classmethod
    def _loose_parse_list(cls, text: str, start: int):
        if start >= len(text) or text[start] != "[":
            raise ValueError("Expected '['")

        idx = start + 1
        result = []
        while idx < len(text):
            idx = cls._skip_ws(text, idx)
            if idx >= len(text):
                break
            if text[idx] == "]":
                return result, idx + 1
            if text[idx] == ",":
                idx += 1
                continue

            value, idx = cls._loose_parse_value(text, idx)
            result.append(value)

            idx = cls._skip_ws(text, idx)
            if idx < len(text) and text[idx] == ",":
                idx += 1
        return result, idx

    @classmethod
    def _loose_parse_value(cls, text: str, start: int):
        idx = cls._skip_ws(text, start)
        if idx >= len(text):
            return None, idx

        ch = text[idx]
        if ch in ('"', "'"):
            value, idx = cls._read_loose_quoted_string(text, idx)
            return value, idx

        if ch == "{":
            value, idx = cls._loose_parse_dict(text, idx)
            return value, idx

        if ch == "[":
            value, idx = cls._loose_parse_list(text, idx)
            return value, idx

        token, idx = cls._read_token(text, idx)
        lowered = token.lower()
        if lowered in {"null", "none"}:
            return None, idx
        if lowered == "true":
            return True, idx
        if lowered == "false":
            return False, idx

        if re.fullmatch(r"-?\d+", token):
            try:
                return int(token), idx
            except ValueError:
                return token, idx
        if re.fullmatch(r"-?\d+\.\d+", token):
            try:
                return float(token), idx
            except ValueError:
                return token, idx
        return token, idx

    @classmethod
    def _skip_ws(cls, text: str, idx: int) -> int:
        while idx < len(text) and text[idx].isspace():
            idx += 1
        return idx

    @classmethod
    def _read_key(cls, text: str, start: int):
        idx = start
        ch = text[idx]
        if ch in ('"', "'"):
            quote = ch
            idx += 1
            buf = []
            while idx < len(text):
                cur = text[idx]
                if cur == "\\" and idx + 1 < len(text):
                    buf.append(text[idx + 1])
                    idx += 2
                    continue
                if cur == quote:
                    idx += 1
                    break
                buf.append(cur)
                idx += 1
            return "".join(buf).strip(), idx

        buf = []
        while idx < len(text):
            cur = text[idx]
            if cur == ":" or cur.isspace() or cur in ",}":
                break
            buf.append(cur)
            idx += 1
        return "".join(buf).strip(), idx

    @classmethod
    def _read_loose_quoted_string(cls, text: str, start: int):
        quote = text[start]
        idx = start + 1
        buf = []
        while idx < len(text):
            cur = text[idx]
            if cur == "\\" and idx + 1 < len(text):
                buf.append(cur)
                buf.append(text[idx + 1])
                idx += 2
                continue
            if cur == quote:
                look = cls._skip_ws(text, idx + 1)
                if look >= len(text) or text[look] in ",}]":
                    return cls._decode_string("".join(buf), quote), idx + 1
                buf.append(cur)
                idx += 1
                continue
            buf.append(cur)
            idx += 1
        return cls._decode_string("".join(buf), quote), idx

    @classmethod
    def _decode_string(cls, raw: str, quote: str) -> str:
        candidate = f"{quote}{raw}{quote}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return raw

    @classmethod
    def _read_token(cls, text: str, start: int):
        idx = start
        buf = []
        while idx < len(text):
            cur = text[idx]
            if cur in ",}]":
                break
            buf.append(cur)
            idx += 1
        return "".join(buf).strip(), idx

    @classmethod
    def _get_enhanced_keys(cls, key_name: str) -> list[str]:
        if not key_name.strip():
            return []
        key_lower = key_name.lower().strip()
        enhanced = [
            key_name,
            key_name.upper(),
            key_name.lower(),
            key_name.capitalize(),
        ]
        mappings = {
            "bbox": [
                "bbox",
                "BBOX",
                "Bbox",
                "box",
                "bounding_box",
                "coordinates",
                "coord",
                "bbox_2d",
            ],
            "label": [
                "label",
                "LABEL",
                "Label",
                "name",
                "title",
                "text",
                "class",
            ],
            "confidence": [
                "confidence",
                "CONFIDENCE",
                "conf",
                "score",
                "probability",
                "prob",
            ],
            "x": ["x", "X", "pos_x", "position_x"],
            "y": ["y", "Y", "pos_y", "position_y"],
            "width": ["width", "w", "W", "WIDTH"],
            "height": ["height", "h", "H", "HEIGHT"],
            "zh": [
                "zh",
                "ZH",
                "chinese",
                "Chinese",
                "CHINESE",
                "中文",
            ],
            "en": [
                "en",
                "EN",
                "english",
                "English",
                "ENGLISH",
                "英文",
                "英语",
            ],
        }
        for base, vars in mappings.items():
            if key_lower == base or key_name in vars:
                enhanced.extend(vars)
                break
        seen = set()
        uniq = []
        for k in enhanced:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        return uniq

    @classmethod
    def _find_key_in_object(cls, obj, key_name: str, precision_match: str):
        if not isinstance(obj, dict) or not key_name.strip():
            return None

        if precision_match == "enabled":
            return obj.get(key_name)

        for k in cls._get_enhanced_keys(key_name):
            if k in obj:
                return obj[k]

        key_lower = key_name.lower()
        for obj_key, obj_value in obj.items():
            if obj_key.lower() == key_lower:
                return obj_value
        return None

    @classmethod
    def _format_value(cls, value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                return ", ".join(map(str, value))
            return str(value)
        return str(value)

    @classmethod
    def _parse_label_filters(cls, label_filter: str) -> list[str]:
        if not label_filter or not label_filter.strip():
            return []
        normalized = label_filter.replace("，", ",")
        return [f.strip() for f in normalized.split(",") if f.strip()]

    @classmethod
    def _matches_label_filter(cls, label_value, filters: list[str]) -> bool:
        if not filters:
            return True
        if not label_value:
            return False
        label_str = str(label_value).lower()
        return any(f.lower() in label_str for f in filters)
