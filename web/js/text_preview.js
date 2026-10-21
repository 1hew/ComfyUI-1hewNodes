import { app } from "../../../scripts/app.js";

const TEXT_PREVIEW_NODES = ["1hew_SaveTxt", "1hew_LoadTxt"];
const PREVIEW_WIDGET_NAME = "text_preview";
const PREVIEW_PROPERTY = "_comfy1hewPreviewText";
const MIN_PREVIEW_HEIGHT = 64;
const MAX_PREVIEW_HEIGHT = 220;
const PREVIEW_LINE_HEIGHT = 21;
const PREVIEW_VERTICAL_PADDING = 58;

const COPY_ICON = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5.25" y="5.25" width="8.5" height="8.5" rx="1.5"/><path d="M10.75 5.25V3.5A1.5 1.5 0 0 0 9.25 2h-5.5A1.5 1.5 0 0 0 2.25 3.5v5.5a1.5 1.5 0 0 0 1.5 1.5h1.5"/></svg>`;

function setCopyPressed(button, pressed) {
    if (!button) {
        return;
    }
    button.style.color = pressed ? "#fbbf24" : "var(--fg-color, #fff)";
    button.style.background = pressed
        ? "rgba(245, 158, 11, 0.2)"
        : (button.matches(":hover") ? "rgba(0, 0, 0, 0.7)" : "rgba(0, 0, 0, 0.5)");
    button.style.boxShadow = pressed
        ? "inset 0 0 0 1px rgba(251, 191, 36, 0.2), 0 0 0 1px rgba(251, 191, 36, 0.16)"
        : "none";
    button.style.transform = pressed ? "scale(0.97)" : "translateY(0)";
    const rect = button.querySelector("svg rect");
    if (rect) {
        rect.style.fill = pressed ? "currentColor" : "none";
        rect.style.fillOpacity = pressed ? "0.18" : "0";
    }
}

function setCopyVisible(state, visible) {
    const button = state?.copyButton;
    if (!button) {
        return;
    }
    const show = Boolean(visible && state.text);
    button.style.opacity = show ? "0.9" : "0";
    button.style.transform = show ? "translateY(0)" : "translateY(3px)";
    button.style.pointerEvents = show ? "auto" : "none";
    button.tabIndex = show ? 0 : -1;
    button.setAttribute("aria-hidden", show ? "false" : "true");
}

function isWideCodePoint(code) {
    return (
        (code >= 0x1100 && code <= 0x115f)
        || (code >= 0x2e80 && code <= 0xa4cf)
        || (code >= 0xac00 && code <= 0xd7a3)
        || (code >= 0xf900 && code <= 0xfaff)
        || (code >= 0xfe30 && code <= 0xfe4f)
        || (code >= 0xff00 && code <= 0xff60)
        || (code >= 0xffe0 && code <= 0xffe6)
    );
}

function visualWidth(line) {
    let width = 0;
    for (const ch of line) {
        width += isWideCodePoint(ch.codePointAt(0)) ? 2 : 1;
    }
    return width;
}

function estimatePreviewHeight(text, nodeWidth) {
    if (!text) {
        return 0;
    }
    const usableWidth = Math.max(120, (Number(nodeWidth) || 240) - 34);
    const charsPerLine = Math.max(12, Math.floor(usableWidth / 7.8));
    let lines = 0;
    for (const line of String(text).split("\n")) {
        lines += Math.max(1, Math.ceil(visualWidth(line) / charsPerLine));
    }
    return Math.min(
        MAX_PREVIEW_HEIGHT,
        Math.max(
            MIN_PREVIEW_HEIGHT,
            lines * PREVIEW_LINE_HEIGHT + PREVIEW_VERTICAL_PADDING,
        ),
    );
}

async function copyTextToClipboard(text) {
    if (!text) {
        return false;
    }
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (error) {
        // Fall through to the legacy path (e.g. clipboard permission denied).
    }
    try {
        const area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        Object.assign(area.style, {
            position: "fixed",
            top: "-1000px",
            left: "-1000px",
            opacity: "0",
        });
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand("copy");
        area.remove();
        return ok;
    } catch (error) {
        return false;
    }
}

function removeLegacyPreviewWidget(node) {
    const widgets = node?.widgets;
    if (!Array.isArray(widgets)) {
        return;
    }
    for (let i = widgets.length - 1; i >= 0; i -= 1) {
        const widget = widgets[i];
        if (widget?.name === "preview_text" && widget?.type !== "textPreview") {
            widgets.splice(i, 1);
        }
    }
}

function installPreviewPointerDrag(node, state) {
    // Nodes 2.0 is handled globally by node_body_drag.js so every preview
    // surface (text, image, video, save nodes) shares the same behavior.
    if (LiteGraph.vueNodesMode) {
        return;
    }
    const content = state.content;
    let drag = null;

    const onPointerDown = (event) => {
        if (event.button !== 0) {
            return;
        }
        if (event.target instanceof Element && event.target.closest("button")) {
            return;
        }
        if (!state.text) {
            return;
        }
        drag = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            nodeX: Number(node.pos?.[0]) || 0,
            nodeY: Number(node.pos?.[1]) || 0,
            moved: false,
        };
        try {
            app.canvas?.selectNode?.(node, false);
        } catch {}
        event.stopPropagation();
        try {
            content.setPointerCapture?.(event.pointerId);
        } catch {}
    };

    const onPointerMove = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) {
            return;
        }
        const dx = event.clientX - drag.startX;
        const dy = event.clientY - drag.startY;
        if (!drag.moved && Math.hypot(dx, dy) < 4) {
            return;
        }
        drag.moved = true;
        event.preventDefault();
        event.stopPropagation();
        const scale = Number(app.canvas?.ds?.scale) || 1;
        node.setPos?.(
            drag.nodeX + dx / scale,
            drag.nodeY + dy / scale,
        );
        content.style.userSelect = "none";
        node.graph?.setDirtyCanvas?.(true, true);
    };

    const finishDrag = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) {
            return;
        }
        const moved = drag.moved;
        drag = null;
        content.style.userSelect = "";
        try {
            content.releasePointerCapture?.(event.pointerId);
        } catch {}
        if (moved) {
            event.preventDefault();
            event.stopPropagation();
            node.graph?.setDirtyCanvas?.(true, true);
        }
    };

    content.addEventListener("pointerdown", onPointerDown);
    content.addEventListener("pointermove", onPointerMove);
    content.addEventListener("pointerup", finishDrag);
    content.addEventListener("pointercancel", finishDrag);
}

function ensurePreview(node) {
    if (!node) {
        return null;
    }
    if (node._comfy1hewTextPreview) {
        return node._comfy1hewTextPreview;
    }
    if (typeof document === "undefined") {
        return null;
    }

    removeLegacyPreviewWidget(node);

    const container = document.createElement("div");
    container.className = "comfy-1hew-text-preview";
    Object.assign(container.style, {
        position: "relative",
        display: "none",
        width: "100%",
        height: "100%",
        minHeight: "0",
        boxSizing: "border-box",
    });

    const content = document.createElement("div");
    content.className = "comfy-1hew-text-preview-content";
    Object.assign(content.style, {
        position: "absolute",
        top: "0",
        right: "0",
        bottom: "0",
        left: "0",
        boxSizing: "border-box",
        padding: "8px 8px 34px",
        overflowY: "auto",
        overflowX: "hidden",
        whiteSpace: "pre-wrap",
        overflowWrap: "anywhere",
        wordBreak: "break-word",
        // 参考 Markdown Note：直接显示在节点背景上，不再画输入框。
        fontFamily: "inherit",
        fontSize: "14px",
        lineHeight: "1.5",
        color: "inherit",
        background: "transparent",
        border: "none",
        borderRadius: "0",
        scrollbarWidth: "thin",
        userSelect: "text",
        cursor: "text",
    });

    container.appendChild(content);

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "comfy-1hew-text-preview-copy";
    copyButton.innerHTML = COPY_ICON;
    copyButton.title = "复制文本";
    copyButton.setAttribute("aria-label", "复制文本");
    copyButton.tabIndex = -1;
    copyButton.setAttribute("aria-hidden", "true");
    Object.assign(copyButton.style, {
        position: "absolute",
        right: "8px",
        bottom: "8px",
        zIndex: "3",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "24px",
        height: "24px",
        minWidth: "24px",
        minHeight: "24px",
        padding: "0",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        borderRadius: "6px",
        background: "rgba(0, 0, 0, 0.5)",
        color: "var(--fg-color, #fff)",
        cursor: "pointer",
        userSelect: "none",
        lineHeight: "0",
        opacity: "0",
        pointerEvents: "none",
        transform: "translateY(3px)",
        transition: "opacity .15s ease, transform .08s ease, background .1s ease, border-color .1s ease, color .1s ease, box-shadow .1s ease",
    });
    copyButton.addEventListener("pointerenter", () => {
        copyButton.style.opacity = "1";
        copyButton.style.background = "rgba(0, 0, 0, 0.7)";
    });
    copyButton.addEventListener("pointerleave", () => {
        copyButton.style.opacity = "0.9";
        copyButton.style.background = "rgba(0, 0, 0, 0.5)";
    });
    container.appendChild(copyButton);

    // Only the wheel event is kept local so the preview can scroll without
    // zooming the canvas. Pointer/click events must bubble to the Vue node
    // root; otherwise the lower preview area cannot select or drag the node.
    container.addEventListener("wheel", (event) => {
        event.stopPropagation();
    }, { passive: true });

    const state = {
        container,
        content,
        text: "",
        height: 0,
        autoHeight: null,
        widget: null,
        copyButton,
        userResized: false,
    };

    const hideCopyIfIdle = () => {
        if (container.matches(":hover") || copyButton.matches(":hover")) {
            return;
        }
        setCopyVisible(state, false);
    };

    container.addEventListener("pointerenter", () => {
        setCopyVisible(state, true);
    });
    container.addEventListener("pointerleave", hideCopyIfIdle);

    const originalMouseEnter = node.onMouseEnter;
    node.onMouseEnter = function () {
        setCopyVisible(state, true);
        return originalMouseEnter?.apply(this, arguments);
    };
    const originalMouseLeave = node.onMouseLeave;
    node.onMouseLeave = function () {
        hideCopyIfIdle();
        return originalMouseLeave?.apply(this, arguments);
    };

    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const result = originalOnResize?.apply(this, arguments);
        const height = Number(size?.[1] ?? this.size?.[1]);
        if (
            state.autoHeight != null
            && Number.isFinite(height)
            && Math.abs(height - state.autoHeight) > 0.5
        ) {
            state.userResized = true;
        }
        return result;
    };

    copyButton.addEventListener("pointerdown", (event) => {
        event.stopPropagation();
        if (event.button !== undefined && event.button !== 0) {
            return;
        }
        setCopyPressed(copyButton, true);
    });
    for (const eventName of ["pointerup", "pointercancel", "pointerleave", "blur"]) {
        copyButton.addEventListener(eventName, () => {
            setCopyPressed(copyButton, false);
        });
    }

    copyButton.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!state.text) {
            return;
        }
        const ok = await copyTextToClipboard(state.text);
        if (!ok) {
            console.warn("[1hewNodes] Failed to copy preview text");
        }
    });
    node._comfy1hewTextPreview = state;

    const widget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget(
            PREVIEW_WIDGET_NAME,
            "div",
            container,
            {
                serialize: false,
                hideOnZoom: false,
                hideInPanel: true,
                getMinHeight: () => (state.text ? MIN_PREVIEW_HEIGHT : 0),
            },
        )
        : null;

    state.widget = widget;
    installPreviewPointerDrag(node, state);
    if (widget) {
        widget.serialize = false;
        widget.options = widget.options || {};
        widget.options.serialize = false;
        widget.options.hideInPanel = true;
        // 交给 LiteGraph 的 growable 布局：预览随节点高度自适应，
        // 不再使用固定 computeSize，否则拖高节点后下方会留空。
        delete widget.computeSize;
    }

    return state;
}

function schedulePreviewResize(node) {
    const run = () => resizeNodeForPreview(node);
    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(run);
    } else {
        setTimeout(run, 0);
    }
}

// ComfyUI 1.55 起 node.size 是 Float64Array，不再是普通 Array，
// 不要用 Array.isArray(node.size) 判断，否则会回退到默认 240 宽并重置节点尺寸。
function getNodeSize(node) {
    const rawWidth = Number(node?.size?.[0]);
    const rawHeight = Number(node?.size?.[1]);
    return [
        Number.isFinite(rawWidth) && rawWidth > 0 ? rawWidth : 240,
        Number.isFinite(rawHeight) && rawHeight >= 0 ? rawHeight : 0,
    ];
}

function resizeNodeForPreview(node) {
    const state = node?._comfy1hewTextPreview;
    if (!state) {
        return;
    }
    if (state.userResized) {
        // 用户手动拖过节点尺寸后，不再自动改变节点高度；预览会随节点高度伸缩。
        node.graph?.setDirtyCanvas?.(true, true);
        return;
    }
    const [width, currentHeight] = getNodeSize(node);
    let nextHeight = null;
    try {
        const computed = node.computeSize?.([width, currentHeight]);
        if (Array.isArray(computed) && Number.isFinite(computed[1])) {
            // computeSize 只按预览的最小高度计算；把期望的预览高度补回去。
            const minPreview = state.text ? MIN_PREVIEW_HEIGHT : 0;
            nextHeight = computed[1] + Math.max(0, state.height) - minPreview;
        }
    } catch (error) {
        console.warn("[1hewNodes] Failed to resize text preview node:", error);
    }

    if (!Number.isFinite(nextHeight)) {
        return;
    }
    if (nextHeight > currentHeight + 0.5) {
        state.autoHeight = nextHeight;
        node.setSize?.([width, nextHeight]);
    } else if (
        state.autoHeight != null
        && currentHeight <= state.autoHeight + 0.5
        && nextHeight < currentHeight - 0.5
    ) {
        // Only shrink space that this preview grew automatically; never
        // collapse a node the user resized by hand.
        state.autoHeight = nextHeight;
        node.setSize?.([width, nextHeight]);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function applyPreviewText(node, text, options = {}) {
    const state = ensurePreview(node);
    if (!state) {
        return;
    }

    const value = text == null ? "" : String(text);
    state.text = value;
    state.content.textContent = value;
    state.content.scrollTop = 0;
    if (state.copyButton) {
        const keepCopyVisible = Number(state.copyButton.style.opacity) > 0;
        state.copyButton.disabled = value.length === 0;
        setCopyPressed(state.copyButton, false);
        setCopyVisible(state, keepCopyVisible);
    }

    const height = estimatePreviewHeight(value, getNodeSize(node)[0]);
    state.height = height;
    state.container.style.display = height > 0 ? "block" : "none";

    if (state.widget) {
        // 不使用固定 computeSize：预览由 growable 布局随节点高度伸缩。
        delete state.widget.computeSize;
        state.widget.options = state.widget.options || {};
        state.widget.options.getMinHeight = () =>
            state.text ? MIN_PREVIEW_HEIGHT : 0;
        delete state.widget.options.getMaxHeight;
        delete state.widget.options.getHeight;
    }

    if (options.persist) {
        node.properties = node.properties || {};
        node.properties[PREVIEW_PROPERTY] = value;
    }

    if (options.resize) {
        schedulePreviewResize(node);
    } else {
        node.graph?.setDirtyCanvas?.(true, true);
    }
}

export function setTextPreview(node, text, options = {}) {
    applyPreviewText(node, text, {
        persist: false,
        resize: true,
        ...options,
    });
}

function toPreviewText(raw) {
    if (raw == null) {
        return null;
    }
    if (Array.isArray(raw)) {
        return raw
            .filter((part) => part != null)
            .map((part) => String(part))
            .join("\n\n");
    }
    return String(raw);
}

function getSavedPreviewText(node, info) {
    const infoValue = info?.properties?.[PREVIEW_PROPERTY];
    if (typeof infoValue === "string") {
        return infoValue;
    }
    const nodeValue = node?.properties?.[PREVIEW_PROPERTY];
    return typeof nodeValue === "string" ? nodeValue : "";
}

app.registerExtension({
    name: "ComfyUI-1hewNodesV3.TextPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const typeName = nodeData?.name || nodeType?.type || nodeType?.title;
        if (!TEXT_PREVIEW_NODES.includes(typeName)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            applyPreviewText(this, getSavedPreviewText(this, null), {
                persist: false,
                resize: false,
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            applyPreviewText(this, getSavedPreviewText(this, info), {
                persist: false,
                resize: true,
            });
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const text = toPreviewText(message?.text ?? message?.output);
            if (text == null) {
                return result;
            }
            applyPreviewText(this, text, { persist: true, resize: true });
            return result;
        };
    },
});
