document.addEventListener("DOMContentLoaded", () => {
    const mjpeg = document.getElementById("video_feed");
    mjpeg.src = "/video_feed";

    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    ws.onopen = () => console.log("WebSocket connected");
    ws.onmessage = (event) => console.log("WS message:", event.data);
    const type = document.getElementById("input");

    const keysDown = {};
    let mouseLook = false;
    let isTyping=false;

    // Keyboard events
    window.addEventListener("keydown", (e) => {
        if (isTyping) return; // ⛔ Ignore game controls while typing

        if (!keysDown[e.key]) {
            keysDown[e.key] = true;
            ws.send(JSON.stringify({ type: "keydown", key: e.key }));
        }

        if (e.key === "m") { // toggle mouse look
            e.preventDefault();
            if (!mouseLook) {
                document.body.requestPointerLock();

            } else {
                document.exitPointerLock();
            }
        }
    });
    type.addEventListener("focus", () => {
        isTyping = true;
        console.log("✍️ Typing mode ON");
    });

    type.addEventListener("blur", () => {
        isTyping = false;
        console.log("⌨️ Typing mode OFF");
    });

    window.addEventListener("keyup", (e) => {
        if (isTyping) return; // ⛔ Ignore while typing
        keysDown[e.key] = false;
        ws.send(JSON.stringify({ type: "keyup", key: e.key }));

    });

    // Mouse look movement
    document.addEventListener("pointerlockchange", () => {
        mouseLook = document.pointerLockElement === document.body;
        console.log("Mouse look:", mouseLook);
    });

    window.addEventListener("mousemove", (e) => {
        if (mouseLook) {
            ws.send(JSON.stringify({ type: "mouse", x: e.movementX, y: e.movementY }));
        }
    });

    // Input prompt via button
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    const clearBtn = document.getElementById("clear");
    const meshBtn = document.getElementById("mesh");
    const animateBtn = document.getElementById("animate");
    const ambianceSlider = document.getElementById("ambiance");
    const luminositySlider = document.getElementById("luminosity");
    const depthSlider = document.getElementById("depth")
    const scaleSlider = document.getElementById("scale")
    const meshToggle = document.getElementById("meshToggle");
    const toggleLabel = document.getElementById("toggleLabel");
    depthValue.textContent = depthSlider.value;
    scaleValue.textContent = scaleSlider.value;
    luminosityValue.textContent = luminositySlider.value;
    ambianceValue.textContent = ambianceSlider.value;
    depthSlider.addEventListener("input", () => {
        depthValue.textContent = depthSlider.value;
    });

    scaleSlider.addEventListener("input", () => {
        scaleValue.textContent = scaleSlider.value;
    });
    luminositySlider.addEventListener("input", () => {
        luminosityValue.textContent = luminositySlider.value;
    });
    ambianceSlider.addEventListener("input", () => {
        ambianceValue.textContent = ambianceSlider.value;
    });

    meshToggle.addEventListener("change", () => {
        const state = meshToggle.checked;
        toggleLabel.textContent = `Mesh: ${state ? "ON" : "OFF"}`;

        // Send to FastAPI
        ws.send(JSON.stringify({
            type: "mesh_toggle",
            value: state
        }));
    });

    // Send prompt
    sendBtn.addEventListener("click", () => {
        const text = input.value.trim();
        if (text.length > 0) {
            ws.send(JSON.stringify({ type: "prompt", text }));
            console.log("Sent prompt:", text);
            input.value = "";
        }
    });
    // Mesh button
    meshBtn.addEventListener("click", () => {
        ws.send(JSON.stringify({ type: "mesh" }));
        console.log("Sent mesh command");
    });
    // Clear button
    clearBtn.addEventListener("click", () => {
        ws.send(JSON.stringify({ type: "clear" }));
        console.log("Sent clear command");
    });


    // Animate button
    animateBtn.addEventListener("click", () => {
        ws.send(JSON.stringify({ type: "animate" }));
        console.log("Sent animate command");
    });

    // Ambiance slider
    ambianceSlider.addEventListener("input", () => {
        ws.send(JSON.stringify({ type: "ambiance", value: ambianceSlider.value }));
        console.log("Ambiance:", ambianceSlider.value);
    });

    // Luminosity slider
    luminositySlider.addEventListener("input", () => {
        ws.send(JSON.stringify({ type: "luminosity", value: luminositySlider.value }));
        console.log("Luminosity:", luminositySlider.value);
    });
    // Mesh Depth slider
    depthSlider.addEventListener("input", () => {
        ws.send(JSON.stringify({ type: "depth", value: depthSlider.value }));
        console.log("Depth:", depthSlider.value);
    });
    // Scale slider
    scaleSlider.addEventListener("input", () => {
        ws.send(JSON.stringify({ type: "scale", value: scaleSlider.value }));
        console.log("Scale:", scaleSlider.value);
    });
});
