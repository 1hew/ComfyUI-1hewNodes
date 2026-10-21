import { app } from "../../../scripts/app.js";

const SAVE_TEXT_NODE = "1hew_SaveTxt";

const FORMAT_VALUES = new Set(["txt", "csv", "md", "json"]);

const DEFAULT_WIDGET_VALUES = {
    filename: "text/ComfyUI",
    format: "txt",
    auto_increment: true,
    save_output: true,
    save_metadata: true,
};

function normalizeNamedValues(input) {
    return {
        filename: typeof input.filename === "string"
            ? input.filename
            : DEFAULT_WIDGET_VALUES.filename,
        format: FORMAT_VALUES.has(input.format)
            ? input.format
            : DEFAULT_WIDGET_VALUES.format,
        auto_increment: typeof input.auto_increment === "boolean"
            ? input.auto_increment
            : DEFAULT_WIDGET_VALUES.auto_increment,
        save_output: typeof input.save_output === "boolean"
            ? input.save_output
            : DEFAULT_WIDGET_VALUES.save_output,
        save_metadata: typeof input.save_metadata === "boolean"
            ? input.save_metadata
            : DEFAULT_WIDGET_VALUES.save_metadata,
    };
}

function positionalValues(named) {
    return [
        named.filename,
        named.format,
        named.auto_increment,
        named.save_output,
        named.save_metadata,
    ];
}

function namedFromSequence(rawValues) {
    if (!Array.isArray(rawValues)) {
        return null;
    }
    let values = rawValues.slice();
    if (values.length === 6) {
        // Layout with encode: filename, format, encode, auto_increment,
        // save_output, save_metadata.
        if (
            FORMAT_VALUES.has(values[1])
            && typeof values[0] === "string"
            && typeof values[2] === "string"
            && typeof values[3] === "boolean"
            && typeof values[4] === "boolean"
            && typeof values[5] === "boolean"
        ) {
            return normalizeNamedValues({
                filename: values[0],
                format: values[1],
                auto_increment: values[3],
                save_output: values[4],
                save_metadata: values[5],
            });
        }
        // Layout with encode and format at the end: filename, encode,
        // auto_increment, save_output, save_metadata, format.
        if (
            FORMAT_VALUES.has(values[5])
            && typeof values[0] === "string"
            && typeof values[1] === "string"
            && typeof values[2] === "boolean"
            && typeof values[3] === "boolean"
            && typeof values[4] === "boolean"
        ) {
            return normalizeNamedValues({
                filename: values[0],
                format: values[5],
                auto_increment: values[2],
                save_output: values[3],
                save_metadata: values[4],
            });
        }
        // Very old workflows may carry a leading preview widget value.
        if (typeof values[0] !== "string" || typeof values[1] !== "string") {
            return null;
        }
        values = values.slice(1);
    }
    if (values.length < 3 || values.length > 5) {
        return null;
    }

    const [filename, second, third, fourth, fifth] = values;
    if (values.length >= 5) {
        if (
            FORMAT_VALUES.has(second)
            && typeof third === "boolean"
            && typeof fourth === "boolean"
            && typeof fifth === "boolean"
        ) {
            // Current order without encode: filename, format, auto_increment,
            // save_output, save_metadata.
            return normalizeNamedValues({
                filename,
                format: second,
                auto_increment: third,
                save_output: fourth,
                save_metadata: fifth,
            });
        }
        if (typeof second === "string") {
            // Older order with encode: filename, encode, auto_increment,
            // save_output, save_metadata.
            return normalizeNamedValues({
                filename,
                auto_increment: third,
                save_output: fourth,
                save_metadata: fifth,
            });
        }
        // Legacy order: filename, auto_increment, save_output, encode,
        // save_metadata.
        return normalizeNamedValues({
            filename,
            auto_increment: second,
            save_output: third,
            save_metadata: fifth,
        });
    }

    if (values.length === 4) {
        if (typeof second === "string") {
            // Current order without encode and save_metadata.
            return normalizeNamedValues({
                filename,
                auto_increment: third,
                save_output: fourth,
            });
        }
        // Legacy order without save_metadata.
        return normalizeNamedValues({
            filename,
            auto_increment: second,
            save_output: third,
        });
    }

    // Oldest layout: filename, auto_increment, save_output.
    return normalizeNamedValues({
        filename,
        auto_increment: second,
        save_output: third,
    });
}

/**
 * Convert Save Text widget values to the current order
 * (filename, format, auto_increment, save_output, save_metadata).
 *
 * Older workflows may still contain an `encode` value and/or format at the
 * end; encode is dropped because Save Text now always writes UTF-8.
 * Missing format defaults to "txt". Newer workflows already carry
 * widgets_values_named and are normalized through it.
 */
export function migrateSaveTextWidgetValues(info) {
    if (!info) {
        return null;
    }

    const namedInput = info.widgets_values_named;
    if (
        namedInput
        && typeof namedInput === "object"
        && (
            Object.prototype.hasOwnProperty.call(namedInput, "format")
            || Object.prototype.hasOwnProperty.call(namedInput, "filename")
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

app.registerExtension({
    name: "ComfyUI-1hewNodes.SaveTextWidgetOrder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const typeName = nodeData?.name || nodeType?.type || nodeType?.title;
        if (typeName !== SAVE_TEXT_NODE) {
            return;
        }

        const originalConfigure = nodeType.prototype.configure;
        if (typeof originalConfigure !== "function") {
            return;
        }

        nodeType.prototype.configure = function (info) {
            migrateSaveTextWidgetValues(info);
            return originalConfigure.apply(this, arguments);
        };
    },
});
