let tfModel;
let canvas, ctx;
let isDrawing = false;
let startX, startY;

window.onload = async () => {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    // Set initial canvas strictly to black
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);
    
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);
    document.getElementById('predictBtn').addEventListener('click', predictDigit);
    
    initBars();

    try {
        console.log("Loading model from /model/model.json...");
        tfModel = await tf.loadLayersModel('./model/model.json');
        console.log("Model loaded!");
        document.getElementById('predictionResult').innerText = "Ready";
    } catch (e) {
        console.error("Error loading model", e);
        document.getElementById('predictionResult').innerText = "Model Load Failed";
    }
};

function startDrawing(e) {
    isDrawing = true;
    e.preventDefault();
    const pos = getPos(e);
    startX = pos.x;
    startY = pos.y;
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getPos(e);
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 18; // thickness suitable for MNIST scaling
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    
    startX = pos.x;
    startY = pos.y;
}

function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
}

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const evt = e.touches ? e.touches[0] : e;
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictionResult').innerText = "-";
    resetBars();
}

function initBars() {
    const container = document.getElementById('confidenceBars');
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const wrap = document.createElement('div');
        wrap.className = 'bar-wrap';
        
        const label = document.createElement('span');
        label.innerText = i.toString();
        label.className = 'bar-label';
        
        const barProg = document.createElement('div');
        barProg.className = 'bar-progress';
        
        const barFill = document.createElement('div');
        barFill.id = 'barFill-' + i;
        barFill.className = 'bar-fill';
        barFill.style.width = '0%';
        
        const barVal = document.createElement('span');
        barVal.id = 'barVal-' + i;
        barVal.className = 'bar-val';
        barVal.innerText = '0%';

        barProg.appendChild(barFill);
        
        wrap.appendChild(label);
        wrap.appendChild(barProg);
        wrap.appendChild(barVal);
        
        container.appendChild(wrap);
    }
}

function resetBars() {
    for (let i = 0; i < 10; i++) {
        document.getElementById('barFill-' + i).style.width = '0%';
        document.getElementById('barVal-' + i).innerText = '0%';
    }
}

async function predictDigit() {
    if (!tfModel) return;
    
    // Preprocessing as required by Notebook (scale down to 28x28, grayscale, normalize [0,1])
    let inputTensor = tf.browser.fromPixels(canvas, 1);
    
    inputTensor = tf.image.resizeBilinear(inputTensor, [28, 28])
                         .toFloat()
                         .div(tf.scalar(255));
    inputTensor = inputTensor.expandDims(0);
    
    const predsTensor = tfModel.predict(inputTensor);
    const predsArray = await predsTensor.data();
    
    let maxIdx = 0;
    let maxVal = -1;
    predsArray.forEach((val, idx) => {
        if (val > maxVal) {
            maxVal = val;
            maxIdx = idx;
        }
        
        const pct = (val * 100).toFixed(1);
        document.getElementById('barFill-' + idx).style.width = pct + '%';
        document.getElementById('barVal-' + idx).innerText = pct + '%';
    });
    
    document.getElementById('predictionResult').innerText = maxIdx;
    
    inputTensor.dispose();
    predsTensor.dispose();
}