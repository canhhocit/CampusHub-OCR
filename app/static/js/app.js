// CampusHub-OCR Testing UI Client Logic (OCR & Face Verification)
document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. Health Checks (OCR & Face Verification)
    // ==========================================
    const ocrEngineStatus = document.getElementById("ocrEngineStatus");
    const faceEngineStatus = document.getElementById("faceEngineStatus");

    function checkOcrHealth() {
        fetch("/api/v1/health")
            .then(res => res.json())
            .then(data => {
                if (data.status === "ok") {
                    ocrEngineStatus.innerHTML = `<span class="status-indicator online"></span> OCR: ${data.ocr_engine} (${data.device})`;
                } else {
                    ocrEngineStatus.innerHTML = `<span class="status-indicator offline"></span> OCR: Không khả dụng`;
                }
            })
            .catch(() => {
                ocrEngineStatus.innerHTML = `<span class="status-indicator offline"></span> OCR: Ngoại tuyến`;
            });
    }

    function checkFaceHealth() {
        fetch("/api/v1/face/health")
            .then(res => res.json())
            .then(data => {
                if (data.status === "ok" && data.models_loaded) {
                    faceEngineStatus.innerHTML = `<span class="status-indicator online"></span> Face: YuNet + SFace (${data.device})`;
                } else if (data.engine_type === "mock") {
                    faceEngineStatus.innerHTML = `<span class="status-indicator warning"></span> Face: Mock Engine (Test)`;
                } else {
                    faceEngineStatus.innerHTML = `<span class="status-indicator offline"></span> Face: Chưa nạp mô hình`;
                }
            })
            .catch(() => {
                faceEngineStatus.innerHTML = `<span class="status-indicator offline"></span> Face: Ngoại tuyến`;
            });
    }

    checkOcrHealth();
    checkFaceHealth();

    // ==========================================
    // 2. Mode Navigation (OCR vs Face Verification)
    // ==========================================
    const modeBtnOcr = document.getElementById("modeBtnOcr");
    const modeBtnFace = document.getElementById("modeBtnFace");
    const viewOcr = document.getElementById("viewOcr");
    const viewFace = document.getElementById("viewFace");
    const useOcrCardBtn = document.getElementById("useOcrCardBtn");

    modeBtnOcr.addEventListener("click", () => switchMode("ocr"));
    modeBtnFace.addEventListener("click", () => switchMode("face"));

    function switchMode(mode) {
        if (mode === "ocr") {
            modeBtnOcr.classList.add("active");
            modeBtnFace.classList.remove("active");
            viewOcr.style.display = "block";
            viewFace.style.display = "none";
        } else {
            modeBtnFace.classList.add("active");
            modeBtnOcr.classList.remove("active");
            viewOcr.style.display = "none";
            viewFace.style.display = "block";

            // If an OCR card is already uploaded, offer 1-click import
            if (currentOcrFile && !currentFaceCardFile) {
                useOcrCardBtn.style.display = "inline-block";
            }
        }
    }

    // ==========================================
    // 3. OCR Module Logic
    // ==========================================
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

    let currentOcrFile = null;
    let latestRawText = "";

    // OCR Tab switching (Structured vs Raw)
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

    // Dropzone interactions for OCR
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
            handleOcrFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleOcrFileSelect(e.target.files[0]);
        }
    });

    function handleOcrFileSelect(file) {
        if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
            showOcrError("Vui lòng chọn tệp hình ảnh định dạng JPG, JPEG hoặc PNG.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showOcrError("Dung lượng tệp vượt quá 10 MB.");
            return;
        }

        currentOcrFile = file;
        hideOcrAlerts();

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.style.display = "none";
            previewContainer.style.display = "block";
            resetBtn.style.display = "inline-block";
            submitBtn.disabled = false;

            const sizeKB = (file.size / 1024).toFixed(1);
            imageMeta.textContent = `${file.name} (${sizeKB} KB)`;

            // Also show button in Face tab to reuse this card
            useOcrCardBtn.style.display = "inline-block";
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        currentOcrFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropzonePrompt.style.display = "block";
        previewContainer.style.display = "none";
        resetBtn.style.display = "none";
        submitBtn.disabled = true;
        statsBadge.style.display = "none";
        hideOcrAlerts();
        resetOcrResults();
    });

    function resetOcrResults() {
        emptyState.style.display = "block";
        tabStructured.style.display = "none";
        tabRaw.style.display = "none";
        fieldsTbody.innerHTML = "";
        rawDetectionsList.innerHTML = "";
        rawCountLabel.textContent = "0 đoạn văn bản được phát hiện";
        latestRawText = "";
    }

    function showOcrError(msg) {
        errorBox.textContent = msg;
        errorBox.style.display = "block";
    }

    function hideOcrAlerts() {
        errorBox.style.display = "none";
        warningBox.style.display = "none";
    }

    submitBtn.addEventListener("click", async () => {
        if (!currentOcrFile) return;

        hideOcrAlerts();
        setOcrLoading(true);

        const formData = new FormData();
        formData.append("file", currentOcrFile);
        formData.append("side", sideSelect.value);
        formData.append("preprocess", preprocessCheck.checked);

        try {
            const response = await fetch("/api/v1/ocr", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                showOcrError(data.detail || "Đã xảy ra lỗi trong quá trình xử lý OCR.");
                resetOcrResults();
                return;
            }

            renderOcrResults(data);
        } catch (err) {
            showOcrError("Không thể kết nối đến máy chủ OCR. Vui lòng kiểm tra dịch vụ.");
            resetOcrResults();
        } finally {
            setOcrLoading(false);
        }
    });

    function setOcrLoading(isLoading) {
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

    function renderOcrResults(data) {
        emptyState.style.display = "none";
        statsBadge.style.display = "flex";
        latencyMs.textContent = data.processing_time_ms;
        detectedSide.textContent = data.side === "FRONT" ? "MẶT TRƯỚC" : "MẶT SAU";

        if (data.warnings && data.warnings.length > 0) {
            warningBox.innerHTML = `<strong>Lưu ý:</strong><br>${data.warnings.join("<br>")}`;
            warningBox.style.display = "block";
        }

        fieldsTbody.innerHTML = "";
        const fields = data.fields;
        const canonicalKeys = data.side === "FRONT"
            ? ["idNumber", "fullName", "dateOfBirth", "gender", "nationality", "placeOfOrigin", "placeOfResidence", "expiryDate"]
            : ["personalFeatures", "issueDate", "issueAuthority", "mrz"];

        canonicalKeys.forEach(key => {
            const field = fields[key];
            if (!field) return;

            const tr = document.createElement("tr");

            const tdLabel = document.createElement("td");
            tdLabel.innerHTML = `
                <div class="field-label">${escapeHtml(field.label)}</div>
                ${field.raw_text ? `<div class="field-raw">Gốc: ${escapeHtml(field.raw_text)}</div>` : ''}
            `;

            const tdVal = document.createElement("td");
            tdVal.className = "field-value";
            if (field.value) {
                tdVal.innerHTML = `
                    <div>${escapeHtml(field.value)}</div>
                    ${field.validation_message ? `<div style="font-size: 11px; color: #d97706; margin-top: 3px;">${escapeHtml(field.validation_message)}</div>` : ''}
                `;
            } else {
                tdVal.innerHTML = `
                    <span style="color: #94a3b8;">—</span>
                    ${field.validation_message ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 3px;">${escapeHtml(field.validation_message)}</div>` : ''}
                `;
            }

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

        const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
        if (activeTab === "structured") {
            tabStructured.style.display = "block";
            tabRaw.style.display = "none";
        } else {
            tabStructured.style.display = "none";
            tabRaw.style.display = "block";
        }
    }

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

    // ==========================================
    // 4. Face Verification Module Logic
    // ==========================================
    const faceCardDropzone = document.getElementById("faceCardDropzone");
    const faceCardFileInput = document.getElementById("faceCardFileInput");
    const faceCardPrompt = document.getElementById("faceCardPrompt");
    const faceCardPreviewContainer = document.getElementById("faceCardPreviewContainer");
    const faceCardPreview = document.getElementById("faceCardPreview");
    const faceCardMeta = document.getElementById("faceCardMeta");
    const changeCardFaceBtn = document.getElementById("changeCardFaceBtn");

    const selfieDropzone = document.getElementById("selfieDropzone");
    const selfieFileInput = document.getElementById("selfieFileInput");
    const selfiePrompt = document.getElementById("selfiePrompt");
    const selfiePreviewContainer = document.getElementById("selfiePreviewContainer");
    const selfiePreview = document.getElementById("selfiePreview");
    const selfieMeta = document.getElementById("selfieMeta");
    const changeSelfieBtn = document.getElementById("changeSelfieBtn");

    const openWebcamBtn = document.getElementById("openWebcamBtn");
    const webcamBox = document.getElementById("webcamBox");
    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const captureWebcamBtn = document.getElementById("captureWebcamBtn");
    const closeWebcamBtn = document.getElementById("closeWebcamBtn");

    const thresholdInput = document.getElementById("thresholdInput");
    const thresholdDisplay = document.getElementById("thresholdDisplay");
    const submitFaceBtn = document.getElementById("submitFaceBtn");
    const resetFaceBtn = document.getElementById("resetFaceBtn");

    const faceEmptyState = document.getElementById("faceEmptyState");
    const faceErrorBox = document.getElementById("faceErrorBox");
    const faceWarningBox = document.getElementById("faceWarningBox");
    const faceResultContent = document.getElementById("faceResultContent");
    const decisionBadge = document.getElementById("decisionBadge");
    const decisionSubtitle = document.getElementById("decisionSubtitle");
    const resSimilarity = document.getElementById("resSimilarity");
    const resThresholdSub = document.getElementById("resThresholdSub");
    const resConfidence = document.getElementById("resConfidence");
    const resLatency = document.getElementById("resLatency");
    const faceDisclaimerBox = document.getElementById("faceDisclaimerBox");

    let currentFaceCardFile = null;
    let currentSelfieFile = null;
    let webcamStream = null;

    // Threshold slider change listener
    thresholdInput.addEventListener("input", (e) => {
        thresholdDisplay.textContent = parseFloat(e.target.value).toFixed(3);
    });

    // 4.1. Card Face Selection
    faceCardDropzone.addEventListener("click", () => {
        if (!currentFaceCardFile) faceCardFileInput.click();
    });

    changeCardFaceBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        faceCardFileInput.click();
    });

    faceCardFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFaceCardSelect(e.target.files[0]);
        }
    });

    useOcrCardBtn.addEventListener("click", () => {
        if (currentOcrFile) {
            handleFaceCardSelect(currentOcrFile);
        }
    });

    function handleFaceCardSelect(file) {
        if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
            showFaceError("Ảnh thẻ CCCD phải thuộc định dạng JPG, JPEG hoặc PNG.");
            return;
        }

        currentFaceCardFile = file;
        hideFaceAlerts();

        const reader = new FileReader();
        reader.onload = (e) => {
            faceCardPreview.src = e.target.result;
            faceCardPrompt.style.display = "none";
            faceCardPreviewContainer.style.display = "block";
            const sizeKB = (file.size / 1024).toFixed(1);
            faceCardMeta.textContent = `${file.name} (${sizeKB} KB)`;
            updateFaceSubmitBtnState();
        };
        reader.readAsDataURL(file);
    }

    // 4.2. Selfie Selection
    selfieDropzone.addEventListener("click", () => {
        if (!currentSelfieFile) selfieFileInput.click();
    });

    changeSelfieBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selfieFileInput.click();
    });

    selfieFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelfieSelect(e.target.files[0]);
        }
    });

    function handleSelfieSelect(file) {
        if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
            showFaceError("Ảnh selfie phải thuộc định dạng JPG, JPEG hoặc PNG.");
            return;
        }

        currentSelfieFile = file;
        hideFaceAlerts();

        // If webcam is active, shut it down
        stopWebcam();

        const reader = new FileReader();
        reader.onload = (e) => {
            selfiePreview.src = e.target.result;
            selfiePrompt.style.display = "none";
            selfiePreviewContainer.style.display = "block";
            const sizeKB = (file.size / 1024).toFixed(1);
            selfieMeta.textContent = `${file.name} (${sizeKB} KB)`;
            updateFaceSubmitBtnState();
        };
        reader.readAsDataURL(file);
    }

    // 4.3. Webcam Capture
    openWebcamBtn.addEventListener("click", async () => {
        hideFaceAlerts();
        try {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                    audio: false
                });
                webcamVideo.srcObject = webcamStream;
                webcamBox.style.display = "flex";
                selfieDropzone.style.display = "none";
            } else {
                showFaceError("Trình duyệt không hỗ trợ truy cập thiết bị webcam.");
            }
        } catch (err) {
            showFaceError("Không thể truy cập camera. Vui lòng cấp quyền webcam hoặc tải ảnh selfie từ máy.");
        }
    });

    closeWebcamBtn.addEventListener("click", () => {
        stopWebcam();
    });

    captureWebcamBtn.addEventListener("click", () => {
        if (!webcamVideo || !webcamStream) return;

        const width = webcamVideo.videoWidth || 640;
        const height = webcamVideo.videoHeight || 480;

        webcamCanvas.width = width;
        webcamCanvas.height = height;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, width, height);

        webcamCanvas.toBlob((blob) => {
            if (!blob) {
                showFaceError("Không thể chụp khung hình từ webcam.");
                return;
            }

            const selfieFile = new File([blob], "selfie_webcam.jpg", { type: "image/jpeg" });
            stopWebcam();
            handleSelfieSelect(selfieFile);
        }, "image/jpeg", 0.92);
    });

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamBox.style.display = "none";
        selfieDropzone.style.display = "block";
    }

    function updateFaceSubmitBtnState() {
        const canSubmit = currentFaceCardFile !== null && currentSelfieFile !== null;
        submitFaceBtn.disabled = !canSubmit;
        resetFaceBtn.style.display = (currentFaceCardFile || currentSelfieFile) ? "inline-block" : "none";
    }

    resetFaceBtn.addEventListener("click", () => {
        currentFaceCardFile = null;
        currentSelfieFile = null;
        faceCardFileInput.value = "";
        selfieFileInput.value = "";
        faceCardPreview.src = "";
        selfiePreview.src = "";
        faceCardPrompt.style.display = "block";
        selfiePrompt.style.display = "block";
        faceCardPreviewContainer.style.display = "none";
        selfiePreviewContainer.style.display = "none";
        stopWebcam();
        hideFaceAlerts();
        resetFaceResults();
        updateFaceSubmitBtnState();
    });

    function resetFaceResults() {
        faceEmptyState.style.display = "block";
        faceResultContent.style.display = "none";
        faceWarningBox.style.display = "none";
    }

    function showFaceError(msg) {
        faceErrorBox.textContent = msg;
        faceErrorBox.style.display = "block";
    }

    function hideFaceAlerts() {
        faceErrorBox.style.display = "none";
        faceWarningBox.style.display = "none";
    }

    // 4.4. Face Verification Request Execution
    submitFaceBtn.addEventListener("click", async () => {
        if (!currentFaceCardFile || !currentSelfieFile) {
            showFaceError("Vui lòng cung cấp cả ảnh mặt trước CCCD và ảnh selfie để thực hiện so khớp.");
            return;
        }

        hideFaceAlerts();
        setFaceLoading(true);

        const formData = new FormData();
        formData.append("card_file", currentFaceCardFile);
        formData.append("selfie_file", currentSelfieFile);
        formData.append("threshold", thresholdInput.value);

        try {
            const response = await fetch("/api/v1/face/verify", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle 503 (model missing) or 400 (bad request)
                const errorMsg = data.detail || `Máy chủ phản hồi mã lỗi ${response.status}.`;
                showFaceError(errorMsg);
                resetFaceResults();
                return;
            }

            renderFaceResults(data);
        } catch (err) {
            showFaceError("Không thể kết nối đến dịch vụ so khớp khuôn mặt. Vui lòng kiểm tra trạng thái máy chủ.");
            resetFaceResults();
        } finally {
            setFaceLoading(false);
        }
    });

    function setFaceLoading(isLoading) {
        submitFaceBtn.disabled = isLoading;
        const spinner = submitFaceBtn.querySelector(".btn-spinner");
        const label = submitFaceBtn.querySelector(".btn-text-label");
        if (isLoading) {
            spinner.style.display = "inline-block";
            label.textContent = "Đang so khớp sinh trắc học...";
        } else {
            spinner.style.display = "none";
            label.textContent = "So khớp khuôn mặt 1:1";
        }
    }

    function renderFaceResults(data) {
        faceEmptyState.style.display = "none";
        faceResultContent.style.display = "block";

        // Decision Badge
        decisionBadge.className = "decision-badge";
        if (data.match === true) {
            decisionBadge.classList.add("match");
            decisionBadge.textContent = "ĐỐI SÁNH KHỚP";
            decisionSubtitle.textContent = "Khuôn mặt trên CCCD và ảnh selfie có độ tương đồng sinh trắc học cao (cùng một người).";
        } else if (!data.success) {
            decisionBadge.classList.add("rejected");
            decisionBadge.textContent = "TỪ CHỐI ĐỐI SÁNH";
            decisionSubtitle.textContent = "Không thể so khớp do phát hiện không thấy khuôn mặt hoặc lỗi chất lượng nghiêm trọng.";
        } else {
            decisionBadge.classList.add("no-match");
            decisionBadge.textContent = "KHÔNG TRÙNG KHỚP";
            decisionSubtitle.textContent = "Độ tương đồng sinh trắc học nằm dưới ngưỡng quyết định.";
        }

        // Metrics
        resSimilarity.textContent = data.similarity_score !== undefined ? data.similarity_score.toFixed(4) : "0.0000";
        resThresholdSub.textContent = `Ngưỡng: ${data.threshold}`;
        resConfidence.textContent = data.confidence_score !== undefined ? `${Math.round(data.confidence_score * 100)}%` : "0%";
        resLatency.textContent = `${data.processing_time_ms} ms`;

        // Quality Analysis Table
        populateQualityTable(data.card_face, data.selfie_face);

        // Warnings list
        if (data.warnings && data.warnings.length > 0) {
            faceWarningBox.innerHTML = `<strong>Cảnh báo chất lượng / Khuyến cáo:</strong><br>${data.warnings.join("<br>")}`;
            faceWarningBox.style.display = "block";
        }

        // Disclaimer
        faceDisclaimerBox.textContent = data.disclaimer || "Kết quả so khớp mang tính chất thử nghiệm thuật toán đối sánh sinh trắc học.";
    }

    function populateQualityTable(card, selfie) {
        // Card face
        const qCardDetected = document.getElementById("qCardDetected");
        const qCardSize = document.getElementById("qCardSize");
        const qCardBrightness = document.getElementById("qCardBrightness");
        const qCardBlur = document.getElementById("qCardBlur");
        const qCardAspect = document.getElementById("qCardAspect");
        const qCardStatus = document.getElementById("qCardStatus");

        if (card && card.box) {
            qCardDetected.innerHTML = `<span class="badge-status badge-VALID">Đã phát hiện (${Math.round(card.box.confidence * 100)}%)</span>`;
            if (card.quality) {
                qCardSize.textContent = `${card.quality.width} x ${card.quality.height} px`;
                qCardBrightness.textContent = `${card.quality.brightness} / 255`;
                qCardBlur.textContent = `${card.quality.blur_score} (Laplacian)`;
                qCardAspect.textContent = `${card.quality.aspect_ratio}`;
                qCardStatus.innerHTML = card.quality.is_valid
                    ? `<span class="badge-status badge-VALID">Đạt tiêu chuẩn</span>`
                    : `<span class="badge-status badge-INVALID">Không đạt</span>`;
            }
        } else {
            qCardDetected.innerHTML = `<span class="badge-status badge-MISSING">${card && card.error_code ? card.error_code : 'Chưa thấy mặt'}</span>`;
            qCardSize.textContent = "—";
            qCardBrightness.textContent = "—";
            qCardBlur.textContent = "—";
            qCardAspect.textContent = "—";
            qCardStatus.textContent = "—";
        }

        // Selfie face
        const qSelfieDetected = document.getElementById("qSelfieDetected");
        const qSelfieSize = document.getElementById("qSelfieSize");
        const qSelfieBrightness = document.getElementById("qSelfieBrightness");
        const qSelfieBlur = document.getElementById("qSelfieBlur");
        const qSelfieAspect = document.getElementById("qSelfieAspect");
        const qSelfieStatus = document.getElementById("qSelfieStatus");

        if (selfie && selfie.box) {
            qSelfieDetected.innerHTML = `<span class="badge-status badge-VALID">Đã phát hiện (${Math.round(selfie.box.confidence * 100)}%)</span>`;
            if (selfie.quality) {
                qSelfieSize.textContent = `${selfie.quality.width} x ${selfie.quality.height} px`;
                qSelfieBrightness.textContent = `${selfie.quality.brightness} / 255`;
                qSelfieBlur.textContent = `${selfie.quality.blur_score} (Laplacian)`;
                qSelfieAspect.textContent = `${selfie.quality.aspect_ratio}`;
                qSelfieStatus.innerHTML = selfie.quality.is_valid
                    ? `<span class="badge-status badge-VALID">Đạt tiêu chuẩn</span>`
                    : `<span class="badge-status badge-INVALID">Không đạt</span>`;
            }
        } else {
            qSelfieDetected.innerHTML = `<span class="badge-status badge-MISSING">${selfie && selfie.error_code ? selfie.error_code : 'Chưa thấy mặt'}</span>`;
            qSelfieSize.textContent = "—";
            qSelfieBrightness.textContent = "—";
            qSelfieBlur.textContent = "—";
            qSelfieAspect.textContent = "—";
            qSelfieStatus.textContent = "—";
        }
    }

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
