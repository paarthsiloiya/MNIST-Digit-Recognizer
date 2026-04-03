let tfModel;
let canvas, ctx;
let isDrawing = false;
let startX, startY;
let predictTimeout;
let brushCursor;
let canvasWrapper;

window.onload = async () => {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    brushCursor = document.getElementById('brushCursor');
    canvasWrapper = document.getElementById('canvasWrapper');

    // Set initial canvas strictly to black
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw event listeners
    canvas.addEventListener('mousedown', startDrawing);
    window.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, {passive: false});      
    window.addEventListener('touchmove', draw, {passive: false});
    window.addEventListener('touchend', stopDrawing);

    // Custom cursor logic
    canvasWrapper.addEventListener('mousemove', updateCursor);
    canvasWrapper.addEventListener('mouseenter', () => brushCursor.classList.remove('hidden'));
    canvasWrapper.addEventListener('mouseleave', () => brushCursor.classList.add('hidden'));

    // Key bind
    document.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() === 'c') clearCanvas();
    });

    document.getElementById('clearBtn').addEventListener('click', clearCanvas); 

    initGrid();

    try {
        const statusEl = document.getElementById('modelStatus');
        tfModel = await tf.loadGraphModel('./model/model.json');
        statusEl.innerHTML = '<i class="ph ph-check-circle"></i> Ready';        
        statusEl.className = 'flex items-center gap-2 text-sm font-medium px-3 py-1 bg-green-50 text-green-700 rounded-full border border-green-200';
    } catch (e) {
        console.error("Error loading model", e);
        const statusEl = document.getElementById('modelStatus');
        statusEl.innerHTML = '<i class="ph ph-warning-circle"></i> Error';      
        statusEl.className = 'flex items-center gap-2 text-sm font-medium px-3 py-1 bg-red-50 text-red-700 rounded-full border border-red-200';
    }
};

function updateCursor(e) {
    const rect = canvasWrapper.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    // Boundaries
    x = Math.max(0, Math.min(x, rect.width));
    y = Math.max(0, Math.min(y, rect.height));

    brushCursor.style.left = x + 'px';
    brushCursor.style.top = y + 'px';
}

function startDrawing(e) {
    if (e.target !== canvas) return;
    isDrawing = true;
    if(e.cancelable) e.preventDefault();

    const pos = getPos(e);
    startX = pos.x;
    startY = pos.y;

    // Draw dot at click
    drawPoint(pos.x, pos.y, e.button === 2 || e.touches && e.touches.length > 1);
}

function draw(e) {
    if (!isDrawing) return;
    if(e.cancelable && e.target === canvas) e.preventDefault();

    const pos = getPos(e);
    const isErase = e.button === 2 || e.buttons === 2; // Right click

    ctx.strokeStyle = isErase ? 'black' : 'white';
    ctx.lineWidth = 1.6; // Soft brush width for 28x28 (approx 1.5-2px)
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    startX = pos.x;
    startY = pos.y;

    debouncePredict();
}

function drawPoint(x, y, isErase) {
    ctx.fillStyle = isErase ? 'black' : 'white';
    ctx.beginPath();
    ctx.arc(x, y, 0.8, 0, Math.PI * 2);
    ctx.fill();
    debouncePredict();
}

function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    debouncePredict();
}

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const evt = e.touches ? e.touches[0] : e;

    // Scale client coordinates to 28x28 canvas logical size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictionResult').innerText = "-";
    document.getElementById('topConfidence').innerText = "--%";
    document.getElementById('predictionHelpText').innerText = "Draw digit to predict.";
    resetGrid();
}

function initGrid() {
    const container = document.getElementById('confidenceGrid');
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const wrap = document.createElement('div');
        wrap.id = 'digitWrap-' + i;
        wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl border border-slate-200 bg-slate-50 overflow-hidden transition-all duration-300';
        
        const fill = document.createElement('div');
        fill.id = 'digitFill-' + i;
        fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-100/50 transition-all duration-300 ease-out z-0';
        fill.style.height = '0%';
        
        const contentWrap = document.createElement('div');
        contentWrap.className = 'relative z-10 flex flex-col items-center pointer-events-none mt-1';
        
        const label = document.createElement('span');
        label.innerText = i.toString();
        label.className = 'text-3xl font-bold text-slate-400 transition-colors duration-300';
        label.id = 'digitLabel-' + i;
        
        const val = document.createElement('span');
        val.id = 'digitVal-' + i;
        val.className = 'text-[11px] font-bold text-slate-400 mt-1 transition-colors duration-300';
        val.innerText = '0%';

        contentWrap.appendChild(label);
        contentWrap.appendChild(val);
        
        wrap.appendChild(fill);
        wrap.appendChild(contentWrap);
        
        container.appendChild(wrap);
    }
}

function resetGrid() {
    for (let i = 0; i < 10; i++) {
        const fill = document.getElementById('digitFill-' + i);
        const valNum = document.getElementById('digitVal-' + i);
        
        if(fill) {
            fill.style.height = '0%';
            fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-100/50 transition-all duration-300 ease-out z-0';
        }
        if(valNum) {
            valNum.innerText = '0%';
            valNum.className = 'text-[11px] font-bold text-slate-400 mt-1 transition-colors duration-300';
        }
        
        const wrap = document.getElementById('digitWrap-' + i);
        const label = document.getElementById('digitLabel-' + i);
        
        if(wrap) wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl border border-slate-200 bg-slate-50 overflow-hidden transition-all duration-300';
        if(label) label.className = 'text-3xl font-bold text-slate-400 transition-colors duration-300';
    }
}

function debouncePredict() {
    clearTimeout(predictTimeout);
    predictTimeout = setTimeout(() => {
        predictDigit();
    }, 250);
}

async function predictDigit() {
    if (!tfModel) return;

    // Extract pixel data directly from 28x28 canvas
    let inputTensor = tf.browser.fromPixels(canvas, 1);
    inputTensor = inputTensor.toFloat().div(tf.scalar(255)).expandDims(0);      

    // Check if canvas is essentially empty (all pixels near 0)
    const sum = inputTensor.sum().dataSync()[0];
    if(sum < 5) { // Threshold for empty drawing
        document.getElementById('predictionResult').innerText = "-";
        document.getElementById('topConfidence').innerText = "--%";
        document.getElementById('predictionHelpText').innerText = "Draw digit to predict.";
        resetGrid();
        inputTensor.dispose();
        return;
    }

    const predsTensor = tfModel.predict(inputTensor);
    const predsArray = await predsTensor.data();

    let maxIdx = 0;
    let maxVal = -1;
    predsArray.forEach((val, idx) => {
        if (val > maxVal) {
            maxVal = val;
            maxIdx = idx;
        }

        const pctRound = Math.round(val * 100);
        
        const fill = document.getElementById('digitFill-' + idx);
        const txt = document.getElementById('digitVal-' + idx);
        const wrap = document.getElementById('digitWrap-' + idx);
        const label = document.getElementById('digitLabel-' + idx);
        
        fill.style.height = pctRound + '%';
        txt.innerText = pctRound + '%';
        
        // Base active styling (non-top prediction)
        wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl border border-slate-200 bg-white overflow-hidden transition-all duration-300 shadow-sm';
        fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-100 transition-all duration-300 ease-out z-0';
        label.className = 'text-3xl font-bold text-slate-700 transition-colors duration-300';
        txt.className = 'text-[11px] font-bold text-slate-500 mt-1 transition-colors duration-300 z-10 bg-white/50 px-1 rounded-md backdrop-blur-sm';
        
        if (idx === maxIdx && val > 0.05) {
            // Highlight top prediction beautifully
            wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl border-2 border-indigo-500 bg-indigo-50 overflow-hidden transition-all duration-300 shadow-md ring-2 ring-indigo-100 ring-offset-2 transform scale-105 z-10';
            fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-500 transition-all duration-300 ease-out z-0';
            label.className = 'text-4xl font-bold text-white transition-colors duration-300 drop-shadow-md pb-1';
            txt.className = 'text-[11px] font-bold text-indigo-700 mt-0 transition-colors duration-300 z-10 bg-white px-2 py-0.5 rounded-md shadow-sm';
        }
    });

    document.getElementById('predictionResult').innerText = maxIdx;
    document.getElementById('topConfidence').innerText = (maxVal * 100).toFixed(1) + '%';
    document.getElementById('predictionHelpText').innerText = "Model is very confident."
    if (maxVal < 0.6) {
        document.getElementById('predictionHelpText').innerText = "Model is unsure."
    }

    inputTensor.dispose();
    predsTensor.dispose();
}
