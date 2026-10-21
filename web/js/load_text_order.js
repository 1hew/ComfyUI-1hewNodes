import { app } from "../../../scripts/app.js";

const LOAD_TEXT_NODE = "1hew_LoadTxt";

function normalizeNamedValues(input) {
    const index = Number(input.index);
    return {
        file: typeof input.file === "string" ? input.file : "",
        index: Number.isFinite(index) ? index : 0,
        include_subdir: typeof input.include_subdir === "boolean"
            ? input.include_subdir
            : false,
    };
}

function positionalValues(named) {
    return [named.file, named.index, named.include_subdir];
}

function namedFromSequence(rawValues) {
    if (!Array.isArray(rawValues)) {
        return null;
    }
    if (rawValues.length === 4) {
        // Legacy order: file, encode, index, include_subdir.
        return normalizeNamedValues({
            file: rawValues[0],
            index: rawValues[2],
            include_subdir: rawValues[3],
        });
    }
    if (rawValues.length === 3) {
        // Current order: file, index, include_subdir.
        return normalizeNamedValues({
            file: rawValues[0],
            index: rawValues[1],
            include_subdir: rawValues[2],
        });
    }
    return null;
}

export function migrateLoadTextWidgetValues(info) {
    if (!info) {
        return null;
    }

    const namedInput = info.widgets_values_named;
    if (
        namedInput
        && typeof namedInput === "object"
        && (
            Object.prototype.hasOwnProperty.call(namedInput, "file")
            || Object.prototype.hasOwnProperty.call(namedInput, "index")
            || Object.prototype.hasOwnProperty.call(namedInput, "encode")
        )
    ) {
        const named = normalizeNamedValues(namedInput);
        info.widgets_values_named = named;
        info.widgets_values = positionalValues(named);
        return named;
    }

    const named = namedFromSequence(info.widgets_values);
    if (!named) {
        return null;
    }
    info.widgets_values_named = named;
    info.widgets_values = positionalValues(named);
    return named;
}

function applyNamedValues(node, named) {
    for (const [name, value] of Object.entries(named)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget) {
            widget.value = value;
        }
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.LoadTextWidgetOrder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== LOAD_TEXT_NODE) {
            return;
        }
        const originalConfigure = nodeType.prototype.configure;
        if (typeof originalConfigure !== "function") {
            return;
        }

        nodeType.prototype.configure = function (info) {
            const named = migrateLoadTextWidgetValues(info);
            const result = originalConfigure.apply(this, arguments);
            // Apply by name after configure as well, so the result is correct
            // regardless of which extension wrapper runs first.
            if (named) {
                applyNamedValues(this, named);
            }
            return result;
        };
    },
});
