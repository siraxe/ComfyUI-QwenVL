import { app } from "/scripts/app.js";

// "Edit prompt presets" button + editor dialog for all QwenVL nodes (HF + GGUF).
// The button is a plain button widget appended after the other widgets, so
// the frontend handles its placement and clicks. Presets are stored server-side via
// the /qwenvl/prompt_presets routes registered by py/AILab_QwenVL_Presets.py.

const TARGET_CLASSES = new Set(["AILab_QwenVL", "AILab_QwenVL_Advanced", "AILab_QwenVL_GGUF", "AILab_QwenVL_GGUF_Advanced"]);
const BTN_LABEL = "✏ Edit preset prompts";
const BTN_NAME = "Edit preset prompts";

/* ------------------------------------------------------------------ */
/* Node button (plain button widget, appended after the other widgets) */
/* ------------------------------------------------------------------ */

function addPresetsButton(node) {
    if (!node.widgets) node.widgets = [];
    if (node.widgets.some((w) => w.name === BTN_NAME)) return;
    const widget = node.addWidget("button", BTN_LABEL, null, () => showPresetEditor(node));
    widget.name = BTN_NAME;
    // Never store the button in widgets_values: it carries no state and would
    // only add noise to the saved workflow.
    widget.serialize = false;
    if (widget.options) widget.options.serialize = false;
    return widget;
}

/* ------------------------------------------------------------------ */
/* Live combo refresh after saving presets                             */
/* ------------------------------------------------------------------ */

async function refreshPresetCombos() {
    const valuesByClass = {};
    for (const cls of TARGET_CLASSES) {
        try {
            const res = await app.api.fetchApi(`/object_info/${encodeURIComponent(cls)}`);
            if (!res.ok) continue;
            const info = await res.json();
            const values = info && info[cls] && info[cls].input &&
                info[cls].input.required && info[cls].input.required.preset_prompt &&
                info[cls].input.required.preset_prompt[0];
            if (Array.isArray(values)) valuesByClass[cls] = values;
        } catch (error) {
            console.warn("[QwenVL.PromptPresets] object_info refresh failed for", cls, error);
        }
    }
    for (const node of (app.graph && app.graph._nodes) || []) {
        if (!TARGET_CLASSES.has(node.comfyClass)) continue;
        const widget = node.widgets && node.widgets.find((w) => w.name === "preset_prompt");
        const values = valuesByClass[node.comfyClass];
        if (widget && values && values.length) {
            widget.options.values = values;
            if (!values.includes(widget.value)) {
                widget.value = values[0];
            }
            if (typeof node.setDirtyCanvas === "function") {
                node.setDirtyCanvas(true, true);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Editor dialog                                                       */
/* ------------------------------------------------------------------ */

const CSS = [
    "dialog.qvl-pp-dialog {",
    "    border: 1px solid #000; border-radius: 0; background: #1a1a1a;",
    "    color: #ddd; box-shadow: 0 2px 6px rgba(0,0,0,0.8);",
    "    font-family: Arial, sans-serif; font-size: 12px; padding: 0;",
    "    width: min(900px, 92vw); height: min(620px, 88vh);",
    "    box-sizing: border-box;",
    "}",
    "dialog.qvl-pp-dialog::backdrop { background: rgba(0,0,0,0.35); }",
    "body.qvl-pp-open > *:not(dialog.qvl-pp-dialog) { filter: blur(5px); }",
    ".qvl-pp-dialog *, .qvl-pp-dialog *::before, .qvl-pp-dialog *::after { box-sizing: inherit; }",
    ".qvl-pp-container { display: flex; flex-direction: column; height: 100%; }",
    ".qvl-pp-title { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px 8px; }",
    ".qvl-pp-title h2 { margin: 0; font-size: 14px; font-weight: bold; color: #ddd; }",
    ".qvl-pp-close { background: transparent; border: 0; color: #999; font-size: 16px; cursor: pointer; padding: 2px 6px; }",
    ".qvl-pp-close:hover { color: #fff; }",
    ".qvl-pp-body { display: flex; flex: 1; min-height: 0; padding: 0 12px; gap: 12px; }",
    ".qvl-pp-left { width: 215px; min-width: 215px; display: flex; flex-direction: column; gap: 10px; min-height: 0; }",
    ".qvl-pp-section { display: flex; flex-direction: column; min-height: 0; }",
    ".qvl-pp-section.user { flex: 1; }",
    ".qvl-pp-section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #8a9a99; padding-bottom: 4px; }",
    ".qvl-pp-list { list-style: none; margin: 0; padding: 0; overflow-y: auto; border: 1px solid #333; background: #222; min-height: 0; }",
    ".qvl-pp-section:not(.user) .qvl-pp-list { max-height: 165px; }",
    ".qvl-pp-list li { padding: 5px 8px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #ccc; border-bottom: 1px solid #2a2a2a; }",
    ".qvl-pp-list li:last-child { border-bottom: 0; }",
    ".qvl-pp-list li:hover { background: #2e2e2e; }",
    ".qvl-pp-list li.selected { background: #3d5452; color: #fff; }",
    ".qvl-pp-list li.qvl-pp-empty { cursor: default; color: #666; }",
    ".qvl-pp-hint { color: #777; font-size: 10.5px; padding-top: 4px; line-height: 1.35; }",
    ".qvl-pp-right { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }",
    ".qvl-pp-header { display: flex; gap: 6px; padding-bottom: 8px; align-items: center; }",
    ".qvl-pp-name { flex: 1; background: #222; border: 1px solid #444; color: #ddd; padding: 5px 8px; font-family: inherit; font-size: 12px; min-width: 0; }",
    ".qvl-pp-name:read-only { color: #888; background: #1e1e1e; }",
    ".qvl-pp-text { flex: 1; min-height: 0; resize: none; background: #222; border: 1px solid #444; color: #ddd; padding: 8px; font-family: Consolas, Menlo, monospace; font-size: 12px; line-height: 1.5; }",
    ".qvl-pp-text:read-only { color: #999; background: #1e1e1e; }",
    ".qvl-pp-text:focus, .qvl-pp-name:focus { outline: none; border-color: #3d5452; }",
    ".qvl-pp-footer { display: flex; align-items: center; justify-content: flex-end; gap: 8px; padding: 10px 12px; }",
    ".qvl-pp-status { margin-right: auto; color: #8a9a99; font-size: 11px; }",
    ".qvl-pp-btn { background: #2f3d3d; border: 1px solid #4a5a5a; color: #ddd; padding: 5px 14px; cursor: pointer; font-family: inherit; font-size: 12px; border-radius: 3px; }",
    ".qvl-pp-btn:hover:not(:disabled) { background: #3d5452; color: #fff; }",
    ".qvl-pp-btn:disabled { opacity: 0.45; cursor: default; }",
    ".qvl-pp-btn.primary { background: #3d6452; border-color: #4a7a62; }",
    ".qvl-pp-btn.primary:hover:not(:disabled) { background: #487a63; }",
    ".qvl-pp-btn.danger:hover:not(:disabled) { background: #6a3d3d; }",
].join("\n");

function injectCss() {
    if (document.getElementById("qvl-pp-css")) return;
    const style = document.createElement("style");
    style.id = "qvl-pp-css";
    style.textContent = CSS;
    document.head.appendChild(style);
}

function el(tag, attrs) {
    const node = document.createElement(tag);
    const children = [];
    for (let i = 2; i < arguments.length; i++) children.push(arguments[i]);
    for (const key of Object.keys(attrs || {})) {
        const value = attrs[key];
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key.slice(0, 2) === "on" && typeof value === "function") {
            node.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value !== null && value !== undefined) {
            node.setAttribute(key, value);
        }
    }
    for (const child of children) {
        if (child === null || child === undefined) continue;
        node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
}

class PresetEditorDialog {
    constructor() {
        injectCss();
        this.defaults = {};
        this.user = {};
        this.selectedKind = null;
        this.selectedName = null;
        this.dirty = false;
        this.buildDom();
    }

    buildDom() {
        this.nameInput = el("input", {
            class: "qvl-pp-name", type: "text", placeholder: "Preset name",
            oninput: () => this.onNameInput(),
        });
        this.textArea = el("textarea", {
            class: "qvl-pp-text", spellcheck: "false", placeholder: "System prompt text...",
            oninput: () => this.onTextInput(),
        });
        this.defaultsList = el("ul", { class: "qvl-pp-list" });
        this.userList = el("ul", { class: "qvl-pp-list" });
        this.status = el("span", { class: "qvl-pp-status", text: "" });

        this.btnNew = el("button", { class: "qvl-pp-btn", text: "New", onclick: () => this.onNew() });
        this.btnDuplicate = el("button", { class: "qvl-pp-btn", text: "Duplicate", onclick: () => this.onDuplicate() });
        this.btnDelete = el("button", { class: "qvl-pp-btn danger", text: "Delete", onclick: () => this.onDelete() });

        const content = el("div", { class: "qvl-pp-container" },
            el("div", { class: "qvl-pp-title" },
                el("h2", { text: "Prompt Presets" }),
                el("button", { class: "qvl-pp-close", text: "✕", title: "Close", onclick: () => this.requestClose() }),
            ),
            el("div", { class: "qvl-pp-body" },
                // Column 1: preset lists
                el("div", { class: "qvl-pp-left" },
                    el("div", { class: "qvl-pp-section" },
                        el("div", { class: "qvl-pp-section-title", text: "Default presets" }),
                        this.defaultsList,
                    ),
                    el("div", { class: "qvl-pp-section user" },
                        el("div", { class: "qvl-pp-section-title", text: "User presets" }),
                        this.userList,
                        el("div", { class: "qvl-pp-hint",
                            text: "Default presets cannot be deleted - use Duplicate to edit a copy. " +
                                  "FL2VA / Ref2VA are special system presets managed by the plugin." }),
                    ),
                ),
                // Column 2: editor
                el("div", { class: "qvl-pp-right" },
                    el("div", { class: "qvl-pp-header" },
                        this.nameInput, this.btnNew, this.btnDuplicate, this.btnDelete,
                    ),
                    this.textArea,
                ),
            ),
            el("div", { class: "qvl-pp-footer" },
                this.status,
                el("button", { class: "qvl-pp-btn primary", text: "Save", onclick: () => this.onSave() }),
                el("button", { class: "qvl-pp-btn", text: "Cancel", onclick: () => this.requestClose() }),
            ),
        );

        this.element = el("dialog", { class: "qvl-pp-dialog" }, content);
        // Clicking the backdrop closes (with a dirty check).
        this.element.addEventListener("click", (event) => {
            if (event.target === this.element) this.requestClose();
        });
        this.element.addEventListener("close", () => {
            document.body.classList.remove("qvl-pp-open");
        });
        document.body.appendChild(this.element);
    }

    async open(currentName) {
        try {
            const res = await app.api.fetchApi("/qwenvl/prompt_presets");
            if (!res.ok) throw new Error("HTTP " + res.status);
            const data = await res.json();
            this.defaults = (data && data.defaults) || {};
            this.user = Object.assign({}, (data && data.user) || {});
        } catch (error) {
            window.alert("[QwenVL] Failed to load prompt presets: " + error);
            return;
        }

        // Preselect the node's current preset when it is a known preset.
        this.selectedKind = null;
        this.selectedName = null;
        if (currentName && Object.prototype.hasOwnProperty.call(this.defaults, currentName)) {
            this.selectedKind = "default";
            this.selectedName = currentName;
        } else if (currentName && Object.prototype.hasOwnProperty.call(this.user, currentName)) {
            this.selectedKind = "user";
            this.selectedName = currentName;
        } else {
            const firstDefault = Object.keys(this.defaults)[0];
            if (firstDefault) {
                this.selectedKind = "default";
                this.selectedName = firstDefault;
            }
        }

        this.dirty = false;
        this.setStatus("");
        this.render();
        document.body.classList.add("qvl-pp-open");
        this.element.showModal();
    }

    setStatus(text) {
        this.status.textContent = text || "";
    }

    render() {
        this.renderList(this.defaultsList, "default");
        this.renderList(this.userList, "user");
        this.renderEditor();
    }

    renderList(listEl, kind) {
        listEl.textContent = "";
        const names = kind === "default" ? Object.keys(this.defaults) : Object.keys(this.user);
        for (const name of names) {
            const li = el("li", { text: name, title: name, onclick: () => this.select(kind, name) });
            if (this.selectedKind === kind && this.selectedName === name) {
                li.classList.add("selected");
            }
            listEl.appendChild(li);
        }
        if (!names.length) {
            listEl.appendChild(el("li", { class: "qvl-pp-empty", text: "No presets yet" }));
        }
    }

    renderEditor() {
        const selected = this.getSelected();
        if (!selected) {
            this.nameInput.value = "";
            this.textArea.value = "";
            this.nameInput.readOnly = true;
            this.textArea.readOnly = true;
            this.btnDuplicate.disabled = true;
            this.btnDelete.disabled = true;
            this.btnNew.disabled = false;
            return;
        }
        const isDefault = selected.kind === "default";
        this.nameInput.value = selected.name;
        this.textArea.value = selected.text;
        this.nameInput.readOnly = isDefault;
        this.textArea.readOnly = isDefault;
        this.btnDuplicate.disabled = false;
        this.btnDelete.disabled = isDefault;
        this.btnNew.disabled = false;
    }

    getSelected() {
        const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
        if (this.selectedKind === "default" && has(this.defaults, this.selectedName)) {
            return { kind: "default", name: this.selectedName, text: this.defaults[this.selectedName] };
        }
        if (this.selectedKind === "user" && has(this.user, this.selectedName)) {
            return { kind: "user", name: this.selectedName, text: this.user[this.selectedName] };
        }
        return null;
    }

    select(kind, name) {
        this.selectedKind = kind;
        this.selectedName = name;
        this.setStatus("");
        this.render();
    }

    uniqueName(base) {
        const taken = new Set(Object.keys(this.defaults).concat(Object.keys(this.user)));
        if (!taken.has(base)) return base;
        let index = 2;
        while (taken.has(base + " " + index)) index += 1;
        return base + " " + index;
    }

    onNew() {
        const name = this.uniqueName("New preset");
        this.user[name] = "";
        this.select("user", name);
        this.dirty = true;
        this.nameInput.focus();
        this.nameInput.select();
    }

    onDuplicate() {
        const selected = this.getSelected();
        if (!selected) return;
        const name = this.uniqueName(selected.name + " (copy)");
        this.user[name] = selected.text;
        this.select("user", name);
        this.dirty = true;
        this.nameInput.focus();
        this.nameInput.select();
    }

    onDelete() {
        if (this.selectedKind !== "user" || !this.selectedName) return;
        if (!window.confirm('Delete user preset "' + this.selectedName + '"?')) return;
        delete this.user[this.selectedName];
        this.dirty = true;
        this.selectedKind = "default";
        this.selectedName = Object.keys(this.defaults)[0] || null;
        this.render();
    }

    onNameInput() {
        if (this.selectedKind !== "user") return;
        const newName = this.nameInput.value.trim();
        const oldName = this.selectedName;
        if (!newName || newName === oldName) return;
        const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
        if (has(this.defaults, newName) || has(this.user, newName)) {
            this.setStatus('"' + newName + '" already exists.');
            return;
        }
        // Rebuild the map to keep ordering with the renamed key.
        const rebuilt = {};
        for (const key of Object.keys(this.user)) {
            rebuilt[key === oldName ? newName : key] = this.user[key];
        }
        this.user = rebuilt;
        this.selectedName = newName;
        this.dirty = true;
        this.setStatus("");
        this.renderList(this.defaultsList, "default");
        this.renderList(this.userList, "user");
        this.nameInput.focus();
        const length = this.nameInput.value.length;
        this.nameInput.setSelectionRange(length, length);
    }

    onTextInput() {
        if (this.selectedKind !== "user") return;
        this.user[this.selectedName] = this.textArea.value;
        this.dirty = true;
    }

    async onSave() {
        // Client-side validation: non-empty names.
        for (const name of Object.keys(this.user)) {
            if (!name.trim()) {
                this.setStatus("Preset names cannot be empty.");
                return;
            }
        }
        this.setStatus("Saving...");
        try {
            const res = await app.api.fetchApi("/qwenvl/prompt_presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user: this.user }),
            });
            if (!res.ok) throw new Error("HTTP " + res.status);
            const data = await res.json();
            this.user = (data && data.user) || {};
        } catch (error) {
            this.setStatus("Save failed: " + error);
            return;
        }
        this.dirty = false;
        this.setStatus("Saved " + Object.keys(this.user).length + " user preset(s).");
        this.render();
        await refreshPresetCombos();
    }

    requestClose() {
        if (this.dirty && !window.confirm("Discard unsaved changes?")) return;
        this.element.close();
    }
}

let dialogInstance = null;

function showPresetEditor(node) {
    if (!dialogInstance) dialogInstance = new PresetEditorDialog();
    const widget = node.widgets && node.widgets.find((w) => w.name === "preset_prompt");
    dialogInstance.open(typeof widget.value === "string" ? widget.value : null);
}

/* ------------------------------------------------------------------ */
/* Extension                                                           */
/* ------------------------------------------------------------------ */

app.registerExtension({
    name: "QwenVL.PromptPresets",
    nodeCreated(node) {
        if (!TARGET_CLASSES.has(node.comfyClass)) return;
        addPresetsButton(node);
    },
});
