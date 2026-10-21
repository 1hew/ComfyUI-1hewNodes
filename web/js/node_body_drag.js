import { app } from "../../../scripts/app.js";

const NODE_PREFIX = "1hew";
const DRAG_THRESHOLD = 4;
const HARD_INTERACTIVE_SELECTOR = [
    "button",
    "input",
    "textarea",
    "select",
    "option",
    "a[href]",
    "[contenteditable='true']",
    "[role='button']",
    "[role='slider']",
    "[role='switch']",
    "[role='combobox']",
    "[role='textbox']",
    ".lg-node-header",
    ".lg-slot",
    "[data-slot]",
    "[data-testid*='slot']",
    "[data-testid*='resize']",
    "[class*='resize-handle']",
    "[class*='resize_handle']",
    "[role='separator']",
].join(",");
const MEDIA_SELECTOR = "audio,video";

function findHeaderNodeAt(x, y) {
    const hits = [];
    for (const host of document.querySelectorAll("[data-node-id]")) {
        const header = host.querySelector(".lg-node-header");
        const rect = header?.getBoundingClientRect?.();
        if (!rect) {
            continue;
        }
        // Include the small badge/title strip just above the header.
        if (
            x < rect.left
            || x > rect.right
            || y < rect.top - 26
            || y > rect.bottom
        ) {
            continue;
        }
        const node = app.graph?.getNodeById?.(host.dataset.nodeId);
        if (!node || !String(node.type || "").startsWith(NODE_PREFIX)) {
            continue;
        }
        hits.push(node);
    }
    return hits.length === 1 ? hits[0] : null;
}

function findNode(target) {
    if (!(target instanceof Element)) {
        return null;
    }
    const host = target.closest("[data-node-id]");
    if (!host) {
        return null;
    }
    const node = app.graph?.getNodeById?.(host.dataset.nodeId);
    if (!node || !String(node.type || "").startsWith(NODE_PREFIX)) {
        return null;
    }
    return { node, host };
}

function install() {
    if (window.__comfy1hewNodeBodyDragInstalled) {
        return;
    }
    window.__comfy1hewNodeBodyDragInstalled = true;

    let drag = null;

    const finish = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) {
            return;
        }
        const moved = drag.moved;
        drag = null;
        document.body.style.userSelect = "";
        if (moved) {
            const suppressClick = (clickEvent) => {
                clickEvent.preventDefault();
                clickEvent.stopPropagation();
            };
            document.addEventListener("click", suppressClick, {
                capture: true,
                once: true,
            });
            setTimeout(() => {
                document.removeEventListener("click", suppressClick, true);
            }, 250);
            event.preventDefault();
            event.stopPropagation();
        }
    };

    document.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || !LiteGraph.vueNodesMode || drag) {
            return;
        }
        if (event.target.closest(HARD_INTERACTIVE_SELECTOR)) {
            return;
        }

        // A newer node can be stacked above an older node and cover its
        // header. Prefer the sole header under the pointer so the lower node
        // is still selectable/draggable at that spot.
        const headerNode = findHeaderNodeAt(event.clientX, event.clientY);
        const found = findNode(event.target);
        if (!headerNode && event.target.closest(MEDIA_SELECTOR)) {
            return;
        }
        const node = headerNode || found?.node;
        if (!node) {
            return;
        }
        drag = {
            node,
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
    }, { capture: true });

    document.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.pointerId) {
            return;
        }
        const dx = event.clientX - drag.startX;
        const dy = event.clientY - drag.startY;
        if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) {
            return;
        }
        drag.moved = true;
        event.preventDefault();
        event.stopPropagation();
        const scale = Number(app.canvas?.ds?.scale) || 1;
        drag.node.setPos?.(
            drag.nodeX + dx / scale,
            drag.nodeY + dy / scale,
        );
        document.body.style.userSelect = "none";
        drag.node.graph?.setDirtyCanvas?.(true, true);
    }, { capture: true });

    document.addEventListener("pointerup", finish, { capture: true });
    document.addEventListener("pointercancel", finish, { capture: true });
}

app.registerExtension({
    name: "ComfyUI-1hewNodes.NodeBodyDrag",
    setup() {
        install();
    },
});
