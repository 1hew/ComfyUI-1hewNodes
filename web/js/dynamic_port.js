import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-1hewNodesV3.DynamicPorts",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const configs = {
            "1hew_MultiStringJoin": { base: "string_", addType: "STRING", select: null, initial: 1 },
            "1hew_MultiImageBatch": { base: "image_", addType: "IMAGE", select: null, initial: 1 },
            "1hew_MultiMaskBatch": { base: "mask_", addType: "MASK", select: null, initial: 1 },
            "1hew_MultiImageStitch": { base: "image_", addType: "IMAGE", select: null, initial: 2 },
            "1hew_ImageMainStitch": { base: "image_", addType: "IMAGE", select: null, initial: 3 },
            "1hew_AnySwitchInt": { base: "input_", addType: "*", select: "select", initial: 1, max: 10 },
            "1hew_ImageListAppend": { base: "image_", addType: "IMAGE", select: null, initial: 2 },
            "1hew_MultiMaskMathOps": { base: "mask_", addType: "MASK", select: null, initial: 2 },
        };
        const typeName = nodeData?.name
            || nodeType?.type
            || nodeType?.title
            || nodeType?.name;
        const cfg = configs[typeName];
        if (!cfg) return;

        const baseInput = cfg.base;
        const selectName = cfg.select;
        const cap = typeof cfg.max === "number" && cfg.max > 0 ? cfg.max : Infinity;

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;

        const enableAutoCompact = typeName === "1hew_AnySwitchInt";

        function compactNodeHeight(node) {
            if (!enableAutoCompact) {
                return;
            }
            if (!node || node.flags?.collapsed) {
                return;
            }

            let desiredHeight = null;
            try {
                const computed = node.computeSize?.([node.size?.[0], node.size?.[1]]);
                if (
                    Array.isArray(computed)
                    && computed.length >= 2
                    && Number.isFinite(computed[1])
                ) {
                    desiredHeight = computed[1];
                }
            } catch (_) {}

            if (
                Number.isFinite(desiredHeight)
                && node.size?.[1] > desiredHeight + 2
            ) {
                node.setSize([node.size[0], desiredHeight]);
                node.setDirtyCanvas(true, true);
            }
        }

        function scheduleCompact(node) {
            if (!enableAutoCompact) {
                return;
            }
            setTimeout(() => compactNodeHeight(node), 0);
            setTimeout(() => compactNodeHeight(node), 120);
            setTimeout(() => compactNodeHeight(node), 600);
        }

        function listDynamicInputs(node) {
            return node.inputs?.filter(inp => inp?.name?.startsWith(baseInput)) || [];
        }

        function nextInputName(node) {
            const inputs = listDynamicInputs(node);
            let maxIndex = 0;
            for (const inp of inputs) {
                const suffix = parseInt(String(inp?.name).slice(baseInput.length), 10);
                if (!isNaN(suffix)) maxIndex = Math.max(maxIndex, suffix);
            }
            return baseInput + (maxIndex + 1);
        }

        function listDynamicInputIndices(node) {
            const idxs = [];
            for (let i = 0; i < (node.inputs?.length || 0); i++) {
                if (node.inputs[i]?.name?.startsWith(baseInput)) idxs.push(i);
            }
            return idxs;
        }

        function isConnectedIndex(node, idx) {
            return (node.inputs?.[idx] ?? {}).link != null;
        }

        function addNextInput(node) {
            const count = listDynamicInputs(node).length;
            if (count >= cap) return;
            const name = nextInputName(node);
            node.addInput(name, cfg.addType);
        }

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

        function normalizeDynamicTypes(node) {
            const dynIdxs = listDynamicInputIndices(node);
            for (let i = 0; i < dynIdxs.length; i++) {
                const idx = dynIdxs[i];
                const slot = node.inputs?.[idx];
                if (slot && slot.type !== cfg.addType) {
                    slot.type = cfg.addType;
                }
            }
        }

        function ensureFirstInput(node) {
            const inputs = listDynamicInputs(node);
            const need = Math.max(1, (cfg.initial ?? 1));
            if (inputs.length === 0) {
                node.addInput(baseInput + "1", cfg.addType);
            }
            let current = listDynamicInputs(node).length;
            const target = Math.min(need, cap);
            while (current < target) {
                addNextInput(node);
                current += 1;
            }
            normalizeDynamicTypes(node);
        }

        function ensureSingleTrailingEmpty(node) {
            const dynIdxs = listDynamicInputIndices(node);
            const minInputs = Math.max(1, (cfg.initial ?? 1));
            const targetMin = Math.min(minInputs, cap);
            if (dynIdxs.length < targetMin) {
                for (let i = dynIdxs.length; i < targetMin; i++) {
                    addNextInput(node);
                }
            }
            let trailingEmpty = 0;
            for (let i = dynIdxs.length - 1; i >= 0; i--) {
                const idx = dynIdxs[i];
                if (!isConnectedIndex(node, idx)) trailingEmpty += 1; else break;
            }
            while (trailingEmpty > 1) {
                const dynIdxs2 = listDynamicInputIndices(node);
                if (dynIdxs2.length <= minInputs) break;
                const lastIdx = dynIdxs2[dynIdxs2.length - 1];
                if (!isConnectedIndex(node, lastIdx)) {
                    node.removeInput(lastIdx);
                    trailingEmpty -= 1;
                } else {
                    break;
                }
            }
            const dynIdxs3 = listDynamicInputIndices(node);
            const lastIdx = dynIdxs3[dynIdxs3.length - 1];
            if (isConnectedIndex(node, lastIdx)) {
                const count = listDynamicInputs(node).length;
                if (count < cap) addNextInput(node);
            }
        }

        function renumberDynamicInputsCompact(node) {
            const dynIdxs = listDynamicInputIndices(node);
            for (let i = 0; i < dynIdxs.length; i++) {
                const idx = dynIdxs[i];
                const expected = baseInput + (i + 1);
                const slot = node.inputs?.[idx];
                if (slot && slot.name !== expected) slot.name = expected;
            }
        }

        function scheduleCompactOnDisconnect(node, slot) {
            if (!node._dpTimers) node._dpTimers = {};
            const key = String(slot);
            clearTimeout(node._dpTimers[key]);
            node._dpTimers[key] = setTimeout(() => {
                const name = node.inputs?.[slot]?.name;
                if (!name || !String(name).startsWith(baseInput)) return;
                const idxs = listDynamicInputIndices(node);
                const pos = idxs.indexOf(slot);
                if (pos < 0) return;
                const empty = !isConnectedIndex(node, slot);
                const laterHasConn = idxs.slice(pos + 1).some(j => isConnectedIndex(node, j));
                if (empty && laterHasConn) {
                    node.removeInput(slot);
                    addNextInput(node);
                    renumberDynamicInputsCompact(node);
                    ensureSingleTrailingEmpty(node);
                    syncSelectRange(node);
                    normalizeDynamicTypes(node);
                    node.setDirtyCanvas(true, true);
                }
            }, 20);
        }

        function syncSelectRange(node) {
            if (!selectName) return;
            const total = listDynamicInputs(node).length;
            if (!node.widgets) return;
            const w = node.widgets.find(w => w.name === selectName);
            if (!w) return;
            if (!w.options) w.options = {};
            w.options.max = Math.max(1, Math.min(total, cap));
            if (typeof w.value === "number" && w.value > w.options.max) {
                w.value = w.options.max;
            }
            node.setDirtyCanvas(true, true);
        }

        nodeType.prototype.onConnectionsChange = function(type, slot, connected, link_info, output) {
            const rv = originalOnConnectionsChange?.call(this, type, slot, connected, link_info, output);
            try {
                if (type !== LiteGraph.INPUT) {
                    syncSelectRange(this);
                    scheduleCompact(this);
                    return rv;
                }
                ensureFirstInput(this);
                const dynIdxs = listDynamicInputIndices(this);
                const isLastDynamic = dynIdxs.length > 0 && dynIdxs[dynIdxs.length - 1] === slot;
                const slotName = this.inputs?.[slot]?.name;
                const isDynamicSlot = typeof slotName === "string" && slotName.startsWith(baseInput);
                if (connected && isDynamicSlot && isLastDynamic) {
                    addNextInput(this);
                }
                if (!connected && isDynamicSlot) {
                    scheduleCompactOnDisconnect(this, slot);
                }
                ensureSingleTrailingEmpty(this);
                syncSelectRange(this);
                normalizeDynamicTypes(this);
                this.setDirtyCanvas(true, true);
                scheduleCompact(this);
            } catch (err) {
                console.error("[1hewNodesV3.DynamicPorts] onConnectionsChange error", err);
            }
            return rv;
        };

        const originalCtor = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function() {
            originalCtor?.call(this);
            try {
                if (this.convertWidgetToInput && this.widgets) {
                    const w = this.widgets.find(w => w.name === baseInput + "1");
                    if (w) this.convertWidgetToInput(w);
                }
                if (selectName && this.widgets && enableAutoCompact) {
                    const w = this.widgets.find(w => w.name === selectName);
                    if (w) {
                        const nodeInstance = this;
                        const prev = w.callback;
                        w.callback = function() {
                            const r = prev?.apply(this, arguments);
                            try {
                                scheduleCompact(nodeInstance);
                            } catch (_) {}
                            return r;
                        };
                    }
                }
                ensureFirstInput(this);
                ensureSingleTrailingEmpty(this);
                syncSelectRange(this);
                normalizeDynamicTypes(this);
                this.setDirtyCanvas(true, true);
            } catch (err) {
                console.error("[1hewNodesV3.DynamicPorts] onAdded error", err);
            }
            try {
                scheduleCompact(this);
            } catch (_) {}
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = originalOnConfigure?.apply(this, arguments);
            try {
                scheduleCompact(this);
            } catch (_) {}
            return r;
        };
    }
});
