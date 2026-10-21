// Parse the trailing metadata block written by the 1hew Save Text node.
// Keep the markers in sync with utils/_text_metadata.py.

export const TEXT_METADATA_BEGIN = "<!-- 1hew_nodes_txt_metadata_begin";
export const TEXT_METADATA_END = "1hew_nodes_txt_metadata_end -->";

export function extractTextMetadata(text) {
    if (typeof text !== "string" || !text) {
        return null;
    }
    const end = text.lastIndexOf(TEXT_METADATA_END);
    if (end < 0) {
        return null;
    }

    // Walk candidate begin markers backwards: workflow values can contain
    // the marker text, so the last begin marker is not necessarily the
    // real block start.
    let searchEnd = text.length;
    while (true) {
        const begin = text.lastIndexOf(TEXT_METADATA_BEGIN, searchEnd - 1);
        if (begin < 0 || begin > end) {
            return null;
        }
        const bodyStart = text.indexOf("\n", begin);
        if (bodyStart >= 0 && bodyStart <= end) {
            const raw = text.slice(bodyStart + 1, end).trim();
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    if (data && typeof data === "object" && !Array.isArray(data)) {
                        return data;
                    }
                } catch {
                    // Not the real block start; try the previous marker.
                }
            }
        }
        searchEnd = begin;
    }
}

export function parseStructuredField(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return value;
    }
    if (typeof value === "string" && value.trim()) {
        try {
            const parsed = JSON.parse(value);
            return parsed && typeof parsed === "object" && !Array.isArray(parsed)
                ? parsed
                : null;
        } catch {
            return null;
        }
    }
    return null;
}
