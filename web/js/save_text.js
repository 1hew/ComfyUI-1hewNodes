import { app } from "../../../scripts/app.js";
import {
    extractTextMetadata,
    parseStructuredField,
} from "./core/text_metadata.js";

// ComfyUI's built-in drag-and-drop reader understands images, video,
// audio, .json and .latent, but not plain .txt/.md files. Save Text embeds
// the workflow in the text/Markdown payload, so wrap handleFile to recognise that
// block; every other file is handed back to the original implementation unchanged.

let installedHandler = null;

function isTextFile(file) {
    if (!file) {
        return false;
    }
    const name = String(file.name || "").toLowerCase();
    return name.endsWith(".txt")
        || name.endsWith(".md")
        || file.type === "text/plain"
        || file.type === "text/markdown";
}

function workflowName(file) {
    return String(file?.name || "text").replace(/\.\w+$/, "");
}

function looksLikeWorkflow(value) {
    return Boolean(value && Array.isArray(value.nodes) && Array.isArray(value.links));
}

async function loadWorkflowFromText(appInstance, file, openSource, options) {
    let text;
    try {
        text = await file.text();
    } catch (error) {
        console.warn("[1hewNodes] Failed to read text file:", error);
        return false;
    }

    const metadata = extractTextMetadata(text);
    if (!metadata) {
        return false;
    }

    const name = workflowName(file);
    const workflow = parseStructuredField(metadata.workflow);
    if (looksLikeWorkflow(workflow)) {
        await appInstance.loadGraphData(workflow, true, true, name, {
            openSource,
            deferWarnings: options?.deferWarnings,
        });
        return true;
    }

    const prompt = parseStructuredField(metadata.prompt);
    if (prompt && typeof appInstance.loadApiJson === "function") {
        await appInstance.loadApiJson(prompt, name, {
            deferWarnings: options?.deferWarnings,
        });
        return true;
    }

    return false;
}

function installTextDropHandler() {
    if (app.handleFile === installedHandler) {
        return true;
    }
    const original = app.handleFile;
    if (typeof original !== "function") {
        return false;
    }

    installedHandler = async function (file, openSource, options) {
        if (isTextFile(file) && typeof file.text === "function") {
            try {
                if (await loadWorkflowFromText(this, file, openSource, options)) {
                    return;
                }
            } catch (error) {
                console.warn(
                    "[1hewNodes] Failed to load workflow from Save Text file:",
                    error,
                );
            }
        }
        return original.call(this, file, openSource, options);
    };
    app.handleFile = installedHandler;
    return true;
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.SaveTextMetadata",
    setup() {
        installTextDropHandler();
    },
});

installTextDropHandler();
