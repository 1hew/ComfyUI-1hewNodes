// ComfyUI-1hewNodes: Dynamic input ports for AnySwitchInt
// This extension enables real-time creation/removal of input_N ports
// and keeps the 'select' widget in sync with available inputs.
// Import ComfyUI app for extension registration
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-1hewNodes.DynamicPorts",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const configs = {
            "1hew_AnySwitchInt": { base: "input_", addType: "*", select: "select", initial: 1 },
            "1hew_TextEncodeQwenImageEditKeepSize": { base: "image_", addType: "IMAGE", select: null, initial: 1 },
            "1hew_MultiStringJoin": { base: "string_", addType: "STRING", select: null, initial: 1 },
            "1hew_MultiImageBatch": { base: "image_", addType: "IMAGE", select: null, initial: 1 },
            "1hew_MultiMaskBatch": { base: "mask_", addType: "MASK", select: null, initial: 1 },
            "1hew_MultiImageStitch": { base: "image_", addType: "IMAGE", select: null, initial: 2 },
        };
        const cfg = configs[nodeData?.name];
        if (!cfg) return;
        try {
            console.log("[1hewNodes.DynamicPorts] attaching to:", nodeData?.name);
        } catch (_) {}

        const baseInput = cfg.base; // input prefix
        const selectName = cfg.select; // widget name controlling selection (optional)

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;

        // Utility: get current dynamic inputs in order
        function listDynamicInputs(node) {
            return node.inputs?.filter(inp => inp?.name?.startsWith(baseInput)) || [];
        }

        // Utility: next dynamic input name
        function nextInputName(node) {
            const inputs = listDynamicInputs(node);
            const n = inputs.length; // existing count
            return baseInput + (n + 1);
        }

        // Utility: count connected dynamic inputs
        function countConnected(node) {
            const inputs = listDynamicInputs(node);
            let c = 0;
            for (const inp of inputs) {
                const idx = indexOfInput(node, inp.name);
                const link = (node.inputs?.[idx] ?? {}).link;
                if (link != null) c += 1;
            }
            return c;
        }

        // Utility: list indices of dynamic inputs
        function listDynamicInputIndices(node) {
            const idxs = [];
            for (let i = 0; i < (node.inputs?.length || 0); i++) {
                if (node.inputs[i]?.name?.startsWith(baseInput)) idxs.push(i);
            }
            return idxs;
        }

        // Utility: check if given input index is connected
        function isConnectedIndex(node, idx) {
            return (node.inputs?.[idx] ?? {}).link != null;
        }

        // Ensure only one empty dynamic input at tail; add one if all connected
        function ensureSingleTrailingEmpty(node) {
            const dynIdxs = listDynamicInputIndices(node);
            const minInputs = Math.max(1, (cfg.initial ?? 1));

            // 至少保证 minInputs 个动态输入存在
            if (dynIdxs.length < minInputs) {
                for (let i = dynIdxs.length; i < minInputs; i++) {
                    addNextInput(node);
                }
            }

            // 统计尾部空槽数量
            let trailingEmpty = 0;
            for (let i = dynIdxs.length - 1; i >= 0; i--) {
                const idx = dynIdxs[i];
                if (!isConnectedIndex(node, idx)) trailingEmpty += 1; else break;
            }

            // 移除多余的尾部空槽：保留至多一个，但不减少到少于 minInputs 个输入
            while (trailingEmpty > 1) {
                const dynIdxs2 = listDynamicInputIndices(node);
                if (dynIdxs2.length <= minInputs) break; // 不少于最小输入数
                const lastIdx = dynIdxs2[dynIdxs2.length - 1];
                if (!isConnectedIndex(node, lastIdx)) {
                    node.removeInput(lastIdx);
                    trailingEmpty -= 1;
                } else {
                    break;
                }
            }

            // 若最后一个槽已连接，则补充一个新的空槽
            const dynIdxs3 = listDynamicInputIndices(node);
            const lastIdx = dynIdxs3[dynIdxs3.length - 1];
            if (isConnectedIndex(node, lastIdx)) {
                addNextInput(node);
            }

            // 重排序号保持 image_1..image_N
            renumberDynamicInputs(node);
        }

        // Renumber dynamic inputs to keep names sequential: input_1..input_N
        function renumberDynamicInputs(node) {
            const dynIdxs = listDynamicInputIndices(node);
            for (let i = 0; i < dynIdxs.length; i++) {
                const idx = dynIdxs[i];
                const expected = baseInput + (i + 1);
                const slot = node.inputs?.[idx];
                if (slot && slot.name !== expected) {
                    slot.name = expected;
                }
            }
        }

        // Normalize: move middle empty slots to tail and keep single empty at tail
        function normalizeDynamicGaps(node) {
            let changed = true;
            while (changed) {
                changed = false;
                const dynIdxs = listDynamicInputIndices(node);
                for (let k = 0; k < dynIdxs.length; k++) {
                    const idx = dynIdxs[k];
                    const isEmpty = !isConnectedIndex(node, idx);
                    const hasLaterConnected = dynIdxs.slice(k + 1).some(j => isConnectedIndex(node, j));
                    // If an empty exists before a later connected, move it to tail
                    if (isEmpty && hasLaterConnected) {
                        node.removeInput(idx);
                        addNextInput(node);
                        changed = true;
                        break; // restart scanning since indices changed
                    }
                }
            }
            ensureSingleTrailingEmpty(node);
            renumberDynamicInputs(node);
        }

        // Update select widget max according to available inputs
        function syncSelectRange(node) {
            if (!selectName) return;
            const total = listDynamicInputs(node).length;
            if (!node.widgets) return;
            const w = node.widgets.find(w => w.name === selectName);
            if (!w) return;
            if (!w.options) w.options = {};
            w.options.max = Math.max(1, total);
            if (typeof w.value === "number" && w.value > w.options.max) {
                w.value = w.options.max;
            }
        }

        // Ensure at least one input_1 exists
        function ensureFirstInput(node) {
            const inputs = listDynamicInputs(node);
            const need = Math.max(1, (cfg.initial ?? 1));
            if (inputs.length === 0) {
                node.addInput(baseInput + "1", cfg.addType);
            }
            // 若需要更多初始输入，依次补齐
            let current = listDynamicInputs(node).length;
            while (current < need) {
                addNextInput(node);
                current += 1;
            }
            renumberDynamicInputs(node);
        }

        // Add new input_N (lazy, wildcard)
        function addNextInput(node) {
            const name = nextInputName(node);
            node.addInput(name, cfg.addType);
        }

        // Helper: find absolute input index by name
        function indexOfInput(node, name) {
            for (let i = 0; i < (node.inputs?.length || 0); i++) {
                if (node.inputs[i]?.name === name) return i;
            }
            return -1;
        }

        // Trim trailing unconnected inputs, keep at least input_1
        function trimTrailing(node) {
            let inputs = listDynamicInputs(node);
            for (let i = inputs.length - 1; i >= 1; i--) {
                const name = inputs[i].name;
                const idx = indexOfInput(node, name);
                const link = (node.inputs?.[idx] ?? {}).link;
                if (link == null) {
                    if (idx >= 0) node.removeInput(idx);
                } else {
                    // stop at first connected from tail
                    break;
                }
            }
        }

        nodeType.prototype.onConnectionsChange = function(type, slot, connected, link_info, output) {
            const rv = originalOnConnectionsChange?.call(this, type, slot, connected, link_info, output);
            try {
                // Only react to input changes
                if (type !== LiteGraph.INPUT) {
                    syncSelectRange(this);
                    return rv;
                }

                ensureFirstInput(this);
                const inputs = listDynamicInputs(this);
                const slotName = this.inputs?.[slot]?.name;
                const isDynamicSlot = typeof slotName === "string" && slotName.startsWith(baseInput);
                const dynIdxs = listDynamicInputIndices(this);
                const isLastDynamic = dynIdxs.length > 0 && dynIdxs[dynIdxs.length - 1] === slot;

                // When connecting to the last dynamic input, always append a new one
                // Align with Impact-Pack: keep one extra empty input at tail
                if (connected && isDynamicSlot && isLastDynamic) {
                    addNextInput(this);
                }

                // Normalize gaps for both connect/disconnect to keep empty only at tail
                normalizeDynamicGaps(this);

                // Sync select range with input count
                syncSelectRange(this);
            } catch (err) {
                console.error("[1hewNodes.DynamicPorts] onConnectionsChange error", err);
            }
            return rv;
        };

        // Initialize when the node is created
        const originalCtor = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function() {
            originalCtor?.call(this);
            try {
                ensureFirstInput(this);
                normalizeDynamicGaps(this);
                syncSelectRange(this);
                console.log("[1hewNodes.DynamicPorts] node initialized");
            } catch (err) {
                console.error("[1hewNodes.DynamicPorts] onAdded error", err);
            }
        };
    }
});