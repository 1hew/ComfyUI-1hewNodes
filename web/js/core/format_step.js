export const VIDEO_FORMATS = {
    "4n+1": { frames: [4, 1] },
    "n": { frames: [1, 0] },
    "2n+1": { frames: [2, 1] },
    "6n+1": { frames: [6, 1] },
    "8n+1": { frames: [8, 1] },
};

export function resolveComboValue(widget, rawValue) {
    if (rawValue === null || rawValue === undefined) {
        return "";
    }
    if (typeof rawValue === "number") {
        const values = widget?.options?.values;
        const candidate = values?.[rawValue];
        return candidate !== undefined && candidate !== null
            ? String(candidate).trim()
            : String(rawValue).trim();
    }
    if (typeof rawValue === "string") {
        const trimmed = rawValue.trim();
        const asNumber = Number(trimmed);
        if (
            Number.isInteger(asNumber) &&
            String(asNumber) === trimmed &&
            widget?.options?.values?.[asNumber] !== undefined
        ) {
            return String(widget.options.values[asNumber]).trim();
        }
        return trimmed;
    }
    return String(rawValue).trim();
}

export function resolveFormatConfig(formatName) {
    const normalized = String(formatName ?? "").trim();
    const direct = VIDEO_FORMATS[normalized];
    if (direct?.frames?.length >= 2) {
        return { step: direct.frames[0], mod: direct.frames[1] };
    }
    const lower = normalized.toLowerCase();
    const lowerDirect = VIDEO_FORMATS[lower];
    if (lowerDirect?.frames?.length >= 2) {
        return { step: lowerDirect.frames[0], mod: lowerDirect.frames[1] };
    }
    if (lower === "" || lower === "n" || lower === "default") {
        return { step: 1, mod: 0 };
    }
    const match = lower.match(/^(\d+)\s*n\s*\+\s*(\d+)$/);
    if (match) {
        return { step: Number(match[1]), mod: Number(match[2]) };
    }
    const nMatch = lower.match(/^(\d+)\s*n$/);
    if (nMatch) {
        return { step: Number(nMatch[1]), mod: 0 };
    }
    return null;
}

