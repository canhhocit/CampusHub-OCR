// CampusHub-OCR Testing UI Client Logic
document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const imageMeta = document.getElementById("imageMeta");
    const submitBtn = document.getElementById("submitBtn");
    const resetBtn = document.getElementById("resetBtn");
    const sideSelect = document.getElementById("sideSelect");
    const preprocessCheck = document.getElementById("preprocessCheck");
    const engineStatus = document.getElementById("engineStatus");

    const emptyState = document.getElementById("emptyState");
    const errorBox = document.getElementById("errorBox");
    const warningBox = document.getElementById("warningBox");
    const tabStructured = document.getElementById("tabStructured");
    const tabRaw = document.getElementById("tabRaw");
    const fieldsTbody = document.getElementById("fieldsTbody");
    const rawDetectionsList = document.getElementById("rawDetectionsList");
    const rawCountLabel = document.getElementById("rawCountLabel");
    const copyRawBtn = document.getElementById("copyRawBtn");
    const statsBadge = document.getElementById("statsBadge");
    const latencyMs = document.getElementById("latencyMs");
    const detectedSide = document.getElementById("detectedSide");
    const tabButtons = document.querySelectorAll(".tab-btn");

    let currentFile = null;
    let latestRawText = "";

    // 1. Health check
    fetch("/api/v1/health")
        .then(res => res.json())
        .then(data => {
            if (data.status === "ok") {
                engineStatus.innerHTML = `<span class="status-indicator online"></span> ${data.ocr_engine} (${data.device}) v${data.version}`;
            }
        })
        .catch(() => {
            engineStatus.innerHTML = `<span class="status-indicator"></span> Ngoại tuyến / Lỗi kết nối`;
        });

    // 2. Tab switching
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const target = btn.dataset.tab;
            if (target === "structured") {
                tabStructured.style.display = "block";
                tabRaw.style.display = "none";
            } else {
                tabStructured.style.display = "none";
                tabRaw.style.display = "block";
            }
        });
    });

    // 3. Dropzone interactions
    dropzone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
            showError("Vui lòng chọn tệp hình ảnh định dạng JPG, JPEG hoặc PNG.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError("Dung lượng tệp vượt quá 10 MB.");
            return;
        }

        currentFile = file;
        hideAlerts();

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.style.display = "none";
            previewContainer.style.display = "block";
            resetBtn.style.display = "inline-block";
            submitBtn.disabled = false;

            const sizeKB = (file.size / 1024).toFixed(1);
            imageMeta.textContent = `${file.name} (${sizeKB} KB)`;
        };
        reader.readAsDataURL(file);
    }

    // 4. Reset
    resetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropzonePrompt.style.display = "block";
        previewContainer.style.display = "none";
        resetBtn.style.display = "none";
        submitBtn.disabled = true;
        statsBadge.style.display = "none";
        hideAlerts();
        resetResults();
    });

    function resetResults() {
        emptyState.style.display = "block";
        tabStructured.style.display = "none";
        tabRaw.style.display = "none";
        fieldsTbody.innerHTML = "";
        rawDetectionsList.innerHTML = "";
        rawCountLabel.textContent = "0 đoạn văn bản được phát hiện";
        latestRawText = "";
    }

    function showError(msg) {
        errorBox.textContent = msg;
        errorBox.style.display = "block";
    }

    function hideAlerts() {
        errorBox.style.display = "none";
        warningBox.style.display = "none";
    }

    // 5. OCR Execution
    submitBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        hideAlerts();
        setLoading(true);

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("side", sideSelect.value);
        formData.append("preprocess", preprocessCheck.checked);

        try {
            const response = await fetch("/api/v1/ocr", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.detail || "Đã xảy ra lỗi trong quá trình xử lý OCR.");
                resetResults();
                return;
            }

            renderResults(data);
        } catch (err) {
            showError("Không thể kết nối đến máy chủ OCR. Vui lòng kiểm tra dịch vụ.");
            resetResults();
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        const spinner = submitBtn.querySelector(".btn-spinner");
        const label = submitBtn.querySelector(".btn-text-label");
        if (isLoading) {
            spinner.style.display = "inline-block";
            label.textContent = "Đang nhận dạng...";
        } else {
            spinner.style.display = "none";
            label.textContent = "Bắt đầu Trích xuất OCR";
        }
    }

    function renderResults(data) {
        emptyState.style.display = "none";
        statsBadge.style.display = "flex";
        latencyMs.textContent = data.processing_time_ms;
        detectedSide.textContent = data.side === "FRONT" ? "MẶT TRƯỚC" : "MẶT SAU";

        // Warnings
        if (data.warnings && data.warnings.length > 0) {
            warningBox.innerHTML = `<strong>Lưu ý:</strong><br>${data.warnings.join("<br>")}`;
            warningBox.style.display = "block";
        }

        // Render Structured Fields
        fieldsTbody.innerHTML = "";
        const fields = data.fields;
        const canonicalKeys = data.side === "FRONT"
            ? ["idNumber", "fullName", "dateOfBirth", "gender", "nationality", "placeOfOrigin", "placeOfResidence", "expiryDate"]
            : ["personalFeatures", "issueDate", "issueAuthority", "mrz"];

        canonicalKeys.forEach(key => {
            const field = fields[key];
            if (!field) return;

            const tr = document.createElement("tr");

            // Label & Raw
            const tdLabel = document.createElement("td");
            tdLabel.innerHTML = `
                <div class="field-label">${escapeHtml(field.label)}</div>
                ${field.raw_text ? `<div class="field-raw">Gốc: ${escapeHtml(field.raw_text)}</div>` : ''}
            `;

            // Value & Validation Warning
            const tdVal = document.createElement("td");
            tdVal.className = "field-value";
            if (field.value) {
                tdVal.innerHTML = `
                    <div>${escapeHtml(field.value)}</div>
                    ${field.validation_message ? `<div style="font-size: 11px; color: #d97706; margin-top: 3px;">⚠️ ${escapeHtml(field.validation_message)}</div>` : ''}
                `;
            } else {
                tdVal.innerHTML = `
                    <span style="color: #94a3b8;">—</span>
                    ${field.validation_message ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 3px;">${escapeHtml(field.validation_message)}</div>` : ''}
                `;
            }

            // Confidence
            const tdConf = document.createElement("td");
            if (field.confidence !== null && field.confidence !== undefined) {
                const pct = Math.round(field.confidence * 100);
                tdConf.innerHTML = `
                    <div class="confidence-bar-wrapper">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${pct}%;"></div>
                        </div>
                        <span class="confidence-val">${pct}%</span>
                    </div>
                `;
            } else {
                tdConf.innerHTML = `<span style="color: #94a3b8;">—</span>`;
            }

            // Status Badge
            const tdStatus = document.createElement("td");
            const statusLabels = {
                EXTRACTED: "HỢP LỆ",
                VALID: "HỢP LỆ",
                NEEDS_REVIEW: "CẦN XEM LẠI",
                MISSING: "CHƯA THẤY",
                NOT_DETECTED: "CHƯA THẤY",
                INVALID: "KHÔNG HỢP LỆ",
                UNREADABLE: "KHÔNG RÕ",
            };
            const labelText = statusLabels[field.status] || field.status;
            tdStatus.innerHTML = `<span class="badge-status badge-${field.status}">${labelText}</span>`;

            tr.appendChild(tdLabel);
            tr.appendChild(tdVal);
            tr.appendChild(tdConf);
            tr.appendChild(tdStatus);
            fieldsTbody.appendChild(tr);
        });

        // Render Raw Detections
        rawDetectionsList.innerHTML = "";
        const detections = data.raw_detections || [];
        rawCountLabel.textContent = `${detections.length} đoạn văn bản được phát hiện`;

        const textLines = [];
        detections.forEach(det => {
            textLines.push(det.text);
            const item = document.createElement("div");
            item.className = "raw-item";
            item.innerHTML = `
                <span class="raw-text">${escapeHtml(det.text)}</span>
                <span class="raw-conf">${Math.round(det.confidence * 100)}%</span>
            `;
            rawDetectionsList.appendChild(item);
        });

        latestRawText = textLines.join("\n");

        // Display current active tab
        const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
        if (activeTab === "structured") {
            tabStructured.style.display = "block";
            tabRaw.style.display = "none";
        } else {
            tabStructured.style.display = "none";
            tabRaw.style.display = "block";
        }
    }

    // Copy raw text
    copyRawBtn.addEventListener("click", () => {
        if (!latestRawText) return;
        navigator.clipboard.writeText(latestRawText).then(() => {
            const originalText = copyRawBtn.textContent;
            copyRawBtn.textContent = "Đã sao chép!";
            setTimeout(() => {
                copyRawBtn.textContent = originalText;
            }, 1800);
        });
    });

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
