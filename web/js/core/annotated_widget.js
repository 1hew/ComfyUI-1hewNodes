import { app } from "../../../../scripts/app.js";
import { resolveComboValue, resolveFormatConfig } from "./format_step.js";

function button_action(widget) {
    if (
        widget.options?.reset == undefined &&
        widget.options?.disable == undefined
    ) {
        return "None";
    }
    if (
        widget.options.reset != undefined &&
        widget.value != widget.options.reset
    ) {
        return "Reset";
    }
    if (
        widget.options.disable != undefined &&
        widget.value != widget.options.disable
    ) {
        return "Disable";
    }
    if (widget.options.reset != undefined) {
        return "No Reset";
    }
    return "No Disable";
}

function fitText(ctx, text, maxLength) {
    if (maxLength <= 0) {
        return ["", 0];
    }
    let fullLength = ctx.measureText(text).width;
    if (fullLength < maxLength) {
        return [text, fullLength];
    }
    let cutoff = ((maxLength / fullLength) * text.length) | 0;
    let shortened = text.slice(0, Math.max(0, cutoff - 2)) + "…";
    return [shortened, ctx.measureText(shortened).width];
}

export function roundToPrecision(num, precision) {
    let strnum = Number(num).toFixed(precision);
    let deci = strnum.indexOf(".");
    if (deci > 0) {
        let i = strnum.length - 1;
        while (i > deci && strnum[i] == "0") {
            i--;
        }
        if (i == deci) {
            i--;
        }
        return strnum.slice(0, i + 1);
    }
    return strnum;
}

function inner_value_change(widget, value, node, pos) {
    // Ensure value is always treated as a number
    widget.value = Number(value);
    if (widget.options?.property && widget.options.property in node.properties) {
        node.setProperty(widget.options.property, widget.value);
    }
    if (widget.callback) {
        widget.callback(widget.value, app.canvas, node, pos);
    }
}

function drawAnnotated(ctx, node, widget_width, y, H) {
    const litegraph_base = LiteGraph;
    const show_text =
        app.canvas.ds.scale >= (app.canvas.low_quality_zoom_threshold ?? 0.5);
    const margin = 15;
    ctx.strokeStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = litegraph_base.WIDGET_BGCOLOR;
    ctx.beginPath();
    if (show_text) {
        ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
    } else {
        ctx.rect(margin, y, widget_width - margin * 2, H);
    }
    ctx.fill();
    if (show_text) {
        if (!this.disabled) {
            ctx.stroke();
        }
        const button = button_action(this);
        if (button != "None") {
            ctx.save();
            if (button.startsWith("No ")) {
                ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
                ctx.strokeStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
            } else {
                ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
                ctx.strokeStyle = litegraph_base.WIDGET_TEXT_COLOR;
            }
            ctx.beginPath();
            if (button.endsWith("Reset")) {
                ctx.arc(
                    widget_width - margin - 26,
                    y + H / 2,
                    4,
                    (Math.PI * 3) / 2,
                    Math.PI
                );
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(widget_width - margin - 26, y + H / 2 - 1.5);
                ctx.lineTo(widget_width - margin - 26, y + H / 2 - 6.5);
                ctx.lineTo(widget_width - margin - 30, y + H / 2 - 3.5);
                ctx.fill();
            } else {
                ctx.arc(
                    widget_width - margin - 26,
                    y + H / 2,
                    4,
                    (Math.PI * 2) / 3,
                    (Math.PI * 8) / 3
                );
                ctx.moveTo(
                    widget_width - margin - 26 - 8 ** 0.5,
                    y + H / 2 + 8 ** 0.5
                );
                ctx.lineTo(
                    widget_width - margin - 26 + 8 ** 0.5,
                    y + H / 2 - 8 ** 0.5
                );
                ctx.stroke();
            }
            ctx.restore();
        }
        ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
        if (!this.disabled) {
            ctx.beginPath();
            ctx.moveTo(margin + 16, y + 5);
            ctx.lineTo(margin + 6, y + H * 0.5);
            ctx.lineTo(margin + 16, y + H - 5);
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(widget_width - margin - 16, y + 5);
            ctx.lineTo(widget_width - margin - 6, y + H * 0.5);
            ctx.lineTo(widget_width - margin - 16, y + H - 5);
            ctx.fill();
        }
        let freeWidth = widget_width - (40 + margin * 2 + 20);
        let [valueText, valueWidth] = fitText(
            ctx,
            this.displayValue?.() ?? this.value ?? "",
            freeWidth
        );
        freeWidth -= valueWidth;

        const value_offset = margin * 2 + 20;
        ctx.textAlign = "left";
        ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
        if (freeWidth > 20) {
            let [name, nameWidth] = fitText(
                ctx,
                this.label || this.name,
                freeWidth
            );
            freeWidth -= nameWidth;
            ctx.fillText(name, margin * 2 + 5, y + H * 0.7);
        }

        let value_offset_right = value_offset;
        ctx.textAlign = "right";
        if (this.options.unit) {
            ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
            let [unitText, unitWidth] = fitText(
                ctx,
                this.options.unit,
                freeWidth
            );
            if (unitText == this.options.unit) {
                ctx.fillText(
                    this.options.unit,
                    widget_width - value_offset_right,
                    y + H * 0.7
                );
                value_offset_right += unitWidth;
                freeWidth -= unitWidth;
            }
        }
        ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
        ctx.fillText(valueText, widget_width - value_offset_right, y + H * 0.7);
        ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;

        let annotation = "";
        if (this.annotation) {
            annotation = this.annotation(this.value, freeWidth);
        } else if (this.options.annotation && this.value in this.options.annotation) {
            annotation = this.options.annotation[this.value];
        }
        if (annotation) {
            ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
            let [annoDisplay] = fitText(ctx, annotation, freeWidth);
            ctx.fillText(
                annoDisplay,
                widget_width - 5 - valueWidth - value_offset_right,
                y + H * 0.7
            );
        }
    }
}

function mouseAnnotated(event, [x, y], node) {
    const now =
        typeof performance !== "undefined" && typeof performance.now === "function"
            ? performance.now()
            : Date.now();
    const duplicateWindowMs = 20;
    const isDownEvent = event.type === "pointerdown" || event.type === "mousedown";
    const isUpEvent = event.type === "pointerup" || event.type === "mouseup";
    if (isDownEvent) {
        if (
            typeof this._comfy1hew_last_down_at === "number" &&
            now - this._comfy1hew_last_down_at < duplicateWindowMs
        ) {
            return true;
        }
        this._comfy1hew_last_down_at = now;
    }
    if (isUpEvent) {
        if (
            typeof this._comfy1hew_last_up_at === "number" &&
            now - this._comfy1hew_last_up_at < duplicateWindowMs
        ) {
            return true;
        }
        this._comfy1hew_last_up_at = now;
    }

    const widget_width = this.width || node.size[0];
    const old_value = this.value;
    const margin = 15;
    let isButton = 0;
    if (x > margin + 6 && x < margin + 16) {
        isButton = -1;
    } else if (x > widget_width - margin - 16 && x < widget_width - margin - 6) {
        isButton = 1;
    } else if (x > widget_width - margin - 34 && x < widget_width - margin - 18) {
        isButton = 2;
    }

    let isAnnotation = false;
    if (this.annotation && !isButton && x > widget_width * 0.5) {
        isAnnotation = true;
    }

    const getStepConfig = () => {
        // Priority 0: Explicit custom config object (most reliable)
        if (this._comfy1hew_step_config) {
            return { 
                step: Number(this._comfy1hew_step_config.step) || 1, 
                mod: Number(this._comfy1hew_step_config.mod) || 0 
            };
        }

        // Priority 1: Custom step resolver (injected by controller)
        if (typeof this._comfy1hew_getStep === "function") {
            const res = this._comfy1hew_getStep();
            if (res && typeof res.step === "number") {
                return { step: res.step, mod: Number(res.mod) || 0 };
            }
        }

        let step = 1;
        let mod = 0;

        // 2. Check widget.step (Standard LiteGraph property)
        // Allow string "4" to be parsed as number 4
        if (this.step !== undefined) {
            const s = Number(this.step);
            if (!isNaN(s) && s > 0) {
                step = s;
            }
        }
        
        // 3. Fallback to options.step
        else if (this.options?.step !== undefined) {
             const s = Number(this.options.step);
             if (!isNaN(s) && s > 0) {
                 step = s;
             }
        }

        // Check for mod in options
        if (this.options?.mod !== undefined) {
            mod = this.options.mod;
        }
        
        // Priority 4: Dynamic lookup from sibling "format" widget (Robust fallback)
        // This covers cases where custom properties are lost or not initialized,
        // matching the logic used in load_video.js's callback.
        if (step === 1) {
            const n = node || this._comfy1hew_owner_node;
            if (n && n.widgets) {
                const formatWidget = n.widgets.find(w => w.name === "format");
                if (formatWidget) {
                    const val = formatWidget.value;
                    const fmtName = resolveComboValue(formatWidget, val);
                    const fmtConfig = resolveFormatConfig(fmtName);
                    if (fmtConfig && fmtConfig.step > 1) {
                        step = fmtConfig.step;
                        mod = fmtConfig.mod;
                    }
                }
            }
        }

        return { step: Number(step), mod: Number(mod) };
    };

    var allow_scroll = true;
    const isDragging =
        typeof event.buttons === "number" ? event.buttons !== 0 : true;
    if (allow_scroll && event.type == "pointermove" && !isButton && isDragging) {
        if (event.deltaX && Math.abs(event.deltaX) > 0.5) {
            const { step, mod } = getStepConfig();
            
            // Apply sensitivity scaling for dragging
            // If step is large (>1), we might want 1:1, but for step=1, 1:1 is too fast.
            // Let's use a standard 0.1 scale for finer control, similar to standard UI sliders.
            const dragScale = 0.1; 
            
            let nextVal = Number(this.value) + event.deltaX * step * dragScale;
            
            // Apply step/mod alignment if step > 1 or mod > 0
            if (step > 1 || mod > 0) {
                if (nextVal <= 0) {
                    nextVal = 0;
                } else {
                    nextVal = Math.round((nextVal - mod) / step) * step + mod;
                    if (nextVal <= 0) {
                        nextVal = 0;
                    }
                }
            }
            
            this.value = nextVal;
        }
        if (this.options.min != null && this.value < this.options.min) {
            this.value = this.options.min;
        }
        if (this.options.max != null && this.value > this.options.max) {
            this.value = this.options.max;
        }
        
        // Final alignment check
        const { step, mod } = getStepConfig();
        if (step > 1 || mod > 0) {
            let alignedVal = Number(this.value);
            if (alignedVal > 0) {
                if (alignedVal < mod) {
                    alignedVal = 0;
                } else {
                    alignedVal = Math.floor((alignedVal - mod) / step) * step + mod;
                    if (alignedVal <= 0) {
                        alignedVal = 0;
                    }
                }
            } else if (alignedVal < 0) {
                alignedVal = 0;
            }
            if (this.options.min != null && alignedVal < this.options.min) {
                alignedVal = this.options.min;
            }
            if (this.options.max != null && alignedVal > this.options.max) {
                alignedVal = this.options.max;
            }
            this.value = alignedVal;
        }
    }
    
    if (event.type == "pointerdown" || event.type == "mousedown") {
        this._mouse_down_on_button = isButton == 2;
        if (isButton == 1 || isButton == -1) {
            this._comfy1hew_arrow_incremented = false;
            
            const { step, mod } = getStepConfig();
            
            // Debug Log for Step Configuration
            console.log("[AnnotatedWidget] Arrow Click", { 
                isButton, 
                step, 
                mod, 
                currentValue: this.value 
            });

            const val = Number(this.value);
            let nextVal = val;

            if (isButton == 1) {
                const s = Number(step) || 1;
                const m = Number(mod) || 0;
                // Calculate distance to next step
                // If current value is 1, step is 4, mod is 1.
                // (1-1)%4 = 0. We want next to be 1+4=5.
                // If current value is 2 (invalid?), (2-1)%4 = 1. d=1. inc = 4-1 = 3. next = 2+3 = 5. Correct.
                let d = ((val - m) % s + s) % s;
                // If d is very close to s, treat as 0
                if (Math.abs(d - s) < 0.001) d = 0;
                
                const inc = d === 0 ? s : s - d;
                nextVal = val + inc;
            } else {
                const s = Number(step) || 1;
                const m = Number(mod) || 0;
                let d = ((val - m) % s + s) % s;
                 // If d is very close to s, treat as 0
                if (Math.abs(d - s) < 0.001) d = 0;
                
                const dec = d === 0 ? s : d;
                nextVal = val - dec;
            }

            // Apply limits
            if (this.options.min != null && nextVal < this.options.min) {
                nextVal = this.options.min;
            }
            if (this.options.max != null && nextVal > this.options.max) {
                nextVal = this.options.max;
            }
            
            // Alignment check after increment/decrement
            if (step > 1 || mod > 0) {
                 if (nextVal > 0) {
                    if (nextVal < mod) {
                        nextVal = 0;
                    } else {
                        nextVal = Math.floor((nextVal - mod) / step) * step + mod;
                        if (nextVal <= 0) {
                            nextVal = 0;
                        }
                    }
                }
            }

            this.value = nextVal;
            this._comfy1hew_arrow_incremented = true;
        } else if (isAnnotation) {
            const anno = this.annotation(this.value, 100);
            if (anno && anno.endsWith("\u21FD")) {
                const numStr = anno.slice(0, -1);
                const val = parseFloat(numStr);
                if (!isNaN(val)) {
                    this.value = val;
                }
            }
        }
    } else if (event.type == "pointerup" || event.type == "mouseup") {
        let buttonClicked = false;

        if (isButton == 2 && (this._mouse_down_on_button || event.click_time < 300)) {
            const buttonType = button_action(this);
            if (buttonType == "Reset") {
                this.value = this.options.reset;
                buttonClicked = true;
            } else if (buttonType == "Disable") {
                this.value = this.options.disable;
                buttonClicked = true;
            }
        }

        if (
            !buttonClicked &&
            event.click_time < 200 &&
            !isButton &&
            !isAnnotation &&
            !this._mouse_down_on_button
        ) {
            const d_callback = (v) => {
                this.value = this.parseValue?.(v) ?? Number(v);
                inner_value_change(this, this.value, node, [x, y]);
            };
            app.canvas.prompt("Value", this.value, d_callback, event);
        }
    }

    this._mouse_down_on_button = false;
    this._comfy1hew_arrow_incremented = false;

    if (old_value != this.value) {
        setTimeout(
            function () {
                inner_value_change(this, this.value, node, [x, y]);
            }.bind(this),
            20
        );
    }
    return true;
}

export function installWidgetSourceOverlay(
    node,
    widget,
    getSourceValue,
    behavior = "reset"
) {
    if (!node || !widget || widget._comfy1hewSourceOverlayInstalled === "1") {
        return;
    }
    widget._comfy1hewSourceOverlayInstalled = "1";
    widget._comfy1hew_owner_node = node;

    widget.draw = drawAnnotated;
    widget.mouse = mouseAnnotated;

    if (behavior === "disable") {
        widget.options.disable = 0;
        delete widget.options.reset;
    } else {
        if (widget.options.reset === undefined) {
            widget.options.reset = widget.options.default ?? widget.value ?? 0;
        }
        delete widget.options.disable;
    }

    widget.annotation = (value, width) => {
        const sourceValue = getSourceValue();
        if (sourceValue !== null && sourceValue !== undefined) {
            return roundToPrecision(sourceValue, 2) + "\u21FD";
        }
        return "";
    };
}

