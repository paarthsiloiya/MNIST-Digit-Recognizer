let tfModel;
let actualKernels = null;
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
        
        try {
            const resp = await fetch('./model/kernels.json');
            actualKernels = await resp.json();
            console.log('Successfully loaded extracted kernels', actualKernels);
        } catch(err) {
            console.warn('Failed to load model/kernels.json. Fallback to mock visuals.', err);
        }

        statusEl.innerHTML = '<i class="ph ph-check-circle"></i> Ready';        
        statusEl.className = 'flex items-center gap-2 text-sm font-medium px-4 py-2 neu-morph-inner text-green-700';
    } catch (e) {
        console.error("Error loading model", e);
        const statusEl = document.getElementById('modelStatus');
        statusEl.innerHTML = '<i class="ph ph-warning-circle"></i> Error';      
        statusEl.className = 'flex items-center gap-2 text-sm font-medium px-4 py-2 neu-morph-inner text-red-600';
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

    // Hide visualizer to return to clean state when fully cleared
    const vizContainer = document.getElementById('vizContainer');
    if (vizContainer) {
        vizContainer.classList.remove('flex');
        vizContainer.classList.add('hidden', 'opacity-0', 'scale-95');
    }
}

function initGrid() {
    const container = document.getElementById('confidenceGrid');
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const wrap = document.createElement('div');
        wrap.id = 'digitWrap-' + i;
        wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] neu-morph-inner rounded-2xl overflow-hidden transition-all duration-300 filter-transition';
        
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
        
        if(wrap) wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] neu-morph-inner rounded-2xl overflow-hidden transition-all duration-300 filter-transition';
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
        wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl neu-morph-inner overflow-hidden transition-all duration-300 filter-transition hover:-translate-y-1';
        fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-100 transition-all duration-300 ease-out z-0';
        label.className = 'text-3xl font-bold text-slate-700 transition-colors duration-300';
        txt.className = 'text-[11px] font-bold text-slate-500 mt-1 transition-colors duration-300 z-10 px-2 py-0.5 rounded-md neu-morph-inner shadow-neu-inner backdrop-blur-sm';
        
        if (idx === maxIdx && val > 0.05) {
            // Highlight top prediction beautifully
            wrap.className = 'relative flex flex-col items-center justify-center aspect-[4/5] rounded-2xl bg-neu-bg overflow-hidden transition-all duration-500 shadow-neu-outer transform scale-105 z-10 filter-transition ring-1 ring-indigo-200';
            fill.className = 'absolute bottom-0 left-0 right-0 w-full bg-indigo-500 transition-all duration-300 ease-out z-0';
            label.className = 'text-4xl font-bold text-white transition-colors duration-300 drop-shadow-md pb-1';
            txt.className = 'text-[11px] font-bold text-indigo-700 mt-0 transition-colors duration-300 z-10 neu-morph-sm px-3 py-1 rounded-lg filter-transition';
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

// --- STBX: Interactive Visualization ---
let vizLayersOutputs = [];
let currentVizLayer = 0;

const layerConfig = [
    { name: 'Conv2D Block 1-A', desc: '32 filters (3x3), extracts simple features.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_1/batchnorm/add_1' },
    { name: 'Conv2D Block 1-B', desc: '32 filters (3x3), refines simple features.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_1_2/batchnorm/add_1' },
    { name: 'Conv2D Block 2-A', desc: '64 filters (3x3), extracts more complex patterns.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_2_1/batchnorm/add_1' },
    { name: 'Conv2D Block 2-B', desc: '64 filters (3x3), further refines patterns.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_3_1/batchnorm/add_1' },
    { name: 'Conv2D Block 3-A', desc: '128 filters (3x3), captures high-level components.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_4_1/batchnorm/add_1' },
    { name: 'Conv2D Block 3-B', desc: '128 filters (3x3), highly abstracted features.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_5_1/batchnorm/add_1' },
    { name: 'Dense 1 (256)', desc: 'First fully connected layer extracting high-level features.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_6_1/batchnorm/add_1' },
    { name: 'Dense 2 (128)', desc: 'Second fully connected layer refining features.', node: 'StatefulPartitionedCall/functional_1/batch_normalization_7_1/batchnorm/add_1' },
    { name: 'Output Layer', desc: 'Final probabilities for each digit (0-9).', node: 'Identity' }
];

async function runVisualizationPrediction(inputTensor) {
    if (!tfModel) return;
    const queryNodes = layerConfig.map(l => l.node);
    try {
        const intermediateOutputs = await tfModel.executeAsync(inputTensor, queryNodes);
        vizLayersOutputs = await Promise.all(intermediateOutputs.map(async (t) => {
            const data = await t.data();
            return { shape: t.shape, data: Array.from(data) };
        }));
        currentVizLayer = 0;
        renderVizLayer();
        document.getElementById('vizContainer').classList.remove('hidden');
    } catch(e) {
        console.warn('Could not extract intermediate layers:', e);
    }
}

function renderVizLayer() {
    if(!vizLayersOutputs.length) return;

    // Update global layer titles and buttons
    const layer = layerConfig[currentVizLayer];
    document.getElementById('vizLayerName').innerText = layer.name;
    document.getElementById('vizLayerDesc').innerText = layer.desc;
    document.getElementById('vizBtnPrev').disabled = currentVizLayer === 0;
    document.getElementById('vizBtnNext').disabled = currentVizLayer === vizLayersOutputs.length - 1;

    // reset currentIntFilter to 0 when changing layers so it doesn't try to access out-of-bounds filters
    if (vizMode === 'interactive' && vizLayersOutputs[currentVizLayer] && vizLayersOutputs[currentVizLayer].shape.length === 4) {
        const shapeChannels = vizLayersOutputs[currentVizLayer].shape[3];
        if (currentIntFilter >= shapeChannels) {
            currentIntFilter = 0;
        }
    }
    
    const isGrid = vizMode === 'grid';
    const gridObj = document.getElementById('vizNodesGrid');
    const intObj = document.getElementById('vizInteractiveContainer');
    const btn = document.getElementById('vizModeToggle');
    
    const layerData = vizLayersOutputs[currentVizLayer];
    const isImageLayer = layerData.shape.length === 4;
    
    if (isGrid || !isImageLayer) {
        gridObj.classList.remove('hidden');
        intObj.classList.add('hidden');
        if (!isImageLayer) {
           btn.style.display = 'none'; // hide toggle if flat
        } else {
           btn.style.display = 'inline-block';
        }
        btn.innerHTML = '<i class="ph ph-swap"></i> Switch to Interactive';
    } else {
        gridObj.classList.add('hidden');
        intObj.classList.remove('hidden');
        btn.innerHTML = '<i class="ph ph-squares-four"></i> Switch to Grid';
        btn.style.display = 'inline-block';
        renderInteractiveCanvas();
        return; // stop execution, we render differently
    }
    const grid = document.getElementById('vizNodesGrid');
    grid.innerHTML = '';
    
    document.getElementById('vizNodesGrid').className = 'flex flex-wrap gap-2.5 max-h-72 overflow-y-auto w-full py-2 px-1 relative z-10 transition-all duration-500 filter-transition custom-scrollbar';

    if (layerData.shape.length === 4) {
        // Conv2D output: [batch, height, width, channels]
        const [batch, h, w, channels] = layerData.shape;
        
        document.getElementById('vizHeaderInfo').innerText = `Filters: ${channels} (${w}x${h})`;
        
        for (let c = 0; c < channels; c++) {
            const div = document.createElement('div');
            div.className = 'w-14 h-14 flex flex-col items-center justify-center neu-morph-sm rounded-xl relative cursor-pointer hover:shadow-neu-hover hover:-translate-y-1 active:shadow-neu-pressed transition-all overflow-hidden';
            
            const vCanvas = document.createElement('canvas');
            vCanvas.width = w;
            vCanvas.height = h;
            vCanvas.className = 'w-full h-full rendering-pixelated block';
            const ctx = vCanvas.getContext('2d');
            const imgData = ctx.createImageData(w, h);
            
            let maxVal = -Infinity;
            let minVal = Infinity;
            
            for(let i=0; i<h; i++) {
                for(let j=0; j<w; j++) {
                    const idx = i * (w * channels) + j * channels + c;
                    const val = layerData.data[idx];
                    if(val > maxVal) maxVal = val;
                    if(val < minVal) minVal = val;
                }
            }
            const range = maxVal - minVal || 1;
            
            for(let i=0; i<h; i++) {
                for(let j=0; j<w; j++) {
                    const idx = i * (w * channels) + j * channels + c;
                    const val = layerData.data[idx];
                    
                    const norm = (val - minVal) / range;
                    const color = Math.floor(norm * 255);
                    const pixelIdx = (i * w + j) * 4;
                    imgData.data[pixelIdx] = color;
                    imgData.data[pixelIdx+1] = color;
                    imgData.data[pixelIdx+2] = color;
                    imgData.data[pixelIdx+3] = 255;
                }
            }
            ctx.putImageData(imgData, 0, 0);
            div.appendChild(vCanvas);
            
            div.title = `Filter ${c}\nShape: ${w}x${h}\nMax Act: ${maxVal.toFixed(3)}`;
            
            div.addEventListener('click', () => {
                currentIntFilter = c;
                vizMode = 'interactive';
                renderVizLayer();
            });
            
            grid.appendChild(div);
        }
    } else {
        // Dense
        const displayData = layerData.data.slice(0, 100);
        const totalNodes = layerData.data.length;
        
        document.getElementById('vizHeaderInfo').innerText = `Nodes showing: ${displayData.length} (of ${totalNodes})`;

        if (totalNodes === 10) {
            // Output Layer specific interactive visualization
            grid.className = 'w-full py-4 relative z-10 transition-all duration-500 filter-transition custom-scrollbar mt-4';
            
            const container = document.createElement('div');
            container.className = 'flex flex-col md:flex-row w-full justify-between items-center gap-8 px-4 h-full';
            
            // Left side: Vertical list
            const leftCol = document.createElement('div');
            leftCol.className = 'flex flex-col gap-2 w-full md:w-1/2';
            
            layerData.data.forEach((val, idx) => {
                const row = document.createElement('div');
                row.className = 'flex items-center gap-3 w-full';
                
                const label = document.createElement('span');
                label.className = 'text-lg font-bold text-slate-700 w-6 text-right';
                label.innerText = idx;
                
                const barTrack = document.createElement('div');
                barTrack.className = 'h-6 flex-1 bg-slate-100 rounded-md overflow-hidden flex';
                
                const barFill = document.createElement('div');
                const intensity = Math.max(0, Math.min(1, val));
                barFill.className = 'h-full transition-all duration-300';
                barFill.style.width = `${intensity * 100}%`;
                barFill.style.backgroundColor = `rgba(79, 70, 229, ${intensity * 0.8 + 0.2})`;
                
                barTrack.appendChild(barFill);
                
                const valText = document.createElement('span');
                valText.className = 'text-xs font-mono text-slate-500 w-12 text-right';
                valText.innerText = val.toFixed(3);
                
                row.appendChild(label);
                row.appendChild(barTrack);
                row.appendChild(valText);
                
                leftCol.appendChild(row);
            });
            
            // Right side: Doughnut chart
            const rightCol = document.createElement('div');
            rightCol.className = 'flex justify-center items-center w-full md:w-1/2 h-48 md:h-64';
            
            const chartCanvas = document.createElement('canvas');
            chartCanvas.className = 'max-h-full max-w-full';
            rightCol.appendChild(chartCanvas);
            
            container.appendChild(leftCol);
            container.appendChild(rightCol);
            grid.appendChild(container);
            
            // Render Chart.js
            const ctx = chartCanvas.getContext('2d');
            const dataColors = Array.from({length: 10}, (_, i) => {
                const intensity = Math.max(0, Math.min(1, layerData.data[i]));
                return `rgba(79, 70, 229, ${intensity * 0.8 + 0.2})`;
            });
            
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['0','1','2','3','4','5','6','7','8','9'],
                    datasets: [{
                        data: layerData.data,
                        backgroundColor: dataColors,
                        borderWidth: 1,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Digit ${context.label}: ${context.raw.toFixed(3)}`;
                                }
                            }
                        }
                    }
                }
            });
            
        } else {
            // Normal Grid for Hidden Dense Layers
            displayData.forEach((val, idx) => {
                const div = document.createElement('div');
                div.className = 'w-10 h-10 flex flex-col items-center justify-center neu-morph-sm rounded-lg text-[10px] relative group cursor-pointer hover:shadow-neu-hover hover:-translate-y-1 active:shadow-neu-pressed transition-all';
                const intensity = Math.min(1, Math.max(0, val));
                div.style.backgroundColor = `rgba(79, 70, 229, ${intensity * 0.4})`;
                const vText = document.createElement('span');
                vText.innerText = val.toFixed(2);
                vText.className = 'select-none pointer-events-none truncate w-full text-center px-1 font-semibold text-slate-700';
                
                div.appendChild(vText);
                
                div.title = `Node ${idx}\nValue: ${val.toFixed(4)}`;
                
                grid.appendChild(div);
            });
        }
    }

}

window.vizPrev = () => { if(currentVizLayer > 0) { currentVizLayer--; renderVizLayer(); } };
window.vizNext = () => { if(currentVizLayer < vizLayersOutputs.length - 1) { currentVizLayer++; renderVizLayer(); } };

const oldPredict = predictDigit;
predictDigit = async function() {
    await oldPredict();
    let inputTensor = tf.browser.fromPixels(canvas, 1).toFloat().div(tf.scalar(255)).expandDims(0);
    const sum = inputTensor.sum().dataSync()[0];
    if(sum >= 5) {
        await runVisualizationPrediction(inputTensor);
    } else {
        document.getElementById('vizContainer').classList.add('hidden');
    }
    inputTensor.dispose();
};

// Interactive Mode State
let vizMode = 'grid'; // 'grid' | 'interactive'
let currentIntFilter = 0;

window.toggleVizMode = () => {
    vizMode = vizMode === 'grid' ? 'interactive' : 'grid';
    renderVizLayer();
};

window.intPrevFilter = () => {
    if(currentIntFilter > 0) {
        currentIntFilter--;
        renderInteractiveCanvas();
    }
};

window.intNextFilter = () => {
    const layerData = vizLayersOutputs[currentVizLayer];
    if (layerData && layerData.shape.length === 4) {
        const channels = layerData.shape[3];
        if (currentIntFilter < channels - 1) {
            currentIntFilter++;
            renderInteractiveCanvas();
        }
    }
};

function renderInteractiveCanvas() {
    const layerData = vizLayersOutputs[currentVizLayer];
    if (!layerData || layerData.shape.length !== 4) return;
    
    const [batch, h, w, channels] = layerData.shape;
    currentIntFilter = Math.min(currentIntFilter, channels - 1);
    
    document.getElementById('intFilterLabel').innerText = 'Filter ' + currentIntFilter;
    document.getElementById('intShapeLabel').innerText = 'Shape: ' + w + 'x' + h;
    
    // Ensure Receptive Field is never empty/broken before hovering immediately loads it
    document.getElementById('rfTargetImg').src = document.getElementById('drawingCanvas').toDataURL();

    const vCanvas = document.getElementById('intCanvas');
    vCanvas.width = w;
    vCanvas.height = h;
    const ctx = vCanvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    
    let maxVal = -Infinity;
    let minVal = Infinity;
    const c = currentIntFilter;
    
    for(let i=0; i<h; i++) {
        for(let j=0; j<w; j++) {
            const idx = i * (w * channels) + j * channels + c;
            const val = layerData.data[idx];
            if(val > maxVal) maxVal = val;
            if(val < minVal) minVal = val;
        }
    }
    const range = maxVal - minVal || 1;
    
    for(let i=0; i<h; i++) {
        for(let j=0; j<w; j++) {
            const idx = i * (w * channels) + j * channels + c;
            const val = layerData.data[idx];
            const norm = (val - minVal) / range;
            const color = Math.floor(norm * 255);
            const pixelIdx = (i * w + j) * 4;
            imgData.data[pixelIdx] = color;
            imgData.data[pixelIdx+1] = color;
            imgData.data[pixelIdx+2] = color;
            imgData.data[pixelIdx+3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // Use actual true kernels if available, else fallback to mock kernel
    let kernelHTML = '';
    if (actualKernels && actualKernels[currentVizLayer] && actualKernels[currentVizLayer][c]) {
        const trueKernel = actualKernels[currentVizLayer][c]; // 3x3 array
        // Flatten to 1D
        const flatKernel = [];
        for (let row of trueKernel) {
            for (let v of row) {
                flatKernel.push(v);
            }
        }
        // Normalize slightly for color display (assuming weights are roughly around -1 to 1)
        let maxAbs = Math.max(...flatKernel.map(Math.abs));
        if (maxAbs === 0) maxAbs = 1;

        for(let i=0; i<9; i++) {
            const kv = flatKernel[i];
            const isPos = kv > 0;
            const alpha = Math.min(1.0, Math.abs(kv) / maxAbs + 0.1); 
            const bg = isPos ? `rgba(79, 70, 229, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
            kernelHTML += `<div style="background-color: ${bg}" class="flex items-center justify-center text-[10px] text-white shadow-inner font-semibold rounded-[2px]" title="Exact Weight: ${kv.toFixed(4)}">${kv.toFixed(2)}</div>`;
        }
    } else {
        // Generate synthetic mock kernel for visual representation (fallback)
        // Static seed since it's now per-filter instead of per-pixel
        const seed = c * 11 + currentVizLayer * 73;
        const randomVal = (i) => (((seed * (i + 1) * 17) % 200) / 100) - 1.0;
        for(let i=0; i<9; i++) {
            const kv = randomVal(i);
            const isPos = kv > 0;
            const bg = isPos ? `rgba(79, 70, 229, ${kv})` : `rgba(239, 68, 68, ${Math.abs(kv)})`;
            kernelHTML += `<div style="background-color: ${bg}" class="flex items-center justify-center text-[10px] text-white shadow-inner font-semibold rounded-[2px]">${kv.toFixed(1)}</div>`;
        }
    }
    document.getElementById('kernelMatrixGrid').innerHTML = kernelHTML;
    
    // Setup hover
    const wrapper = document.getElementById('intCanvasWrapper');
    wrapper.onmousemove = (e) => {
        const rect = vCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const pw = rect.width / w;
        const ph = rect.height / h;
        
        const gridX = Math.floor(x / pw);
        const gridY = Math.floor(y / ph);
        
        // Boundaries
        if(gridX < 0 || gridX >= w || gridY < 0 || gridY >= h) return;
        
        const hl = document.getElementById('intHighlight');
        hl.style.width = pw + 'px';
        hl.style.height = ph + 'px';
        hl.style.left = (gridX * pw) + 'px';
        hl.style.top = (gridY * ph) + 'px';
        hl.classList.remove('hidden');
        
        // Exact value
        const idx = gridY * (w * channels) + gridX * channels + c;
        const val = layerData.data[idx];
        
        // Calculate Receptive field mapping back to the 28x28 input canvas
        let scale = 1, rf_size = 3;
        if (currentVizLayer === 1) { rf_size = 5; }
        else if (currentVizLayer === 2) { scale = 2; rf_size = 10; }
        else if (currentVizLayer === 3) { scale = 2; rf_size = 14; }
        else if (currentVizLayer === 4) { scale = 4; rf_size = 24; }
        else if (currentVizLayer === 5) { scale = 4; rf_size = 32; }

        const centerOriginalX = gridX * scale + Math.floor(scale / 2);
        const centerOriginalY = gridY * scale + Math.floor(scale / 2);
        let topLeftX = centerOriginalX - Math.floor(rf_size / 2);
        let topLeftY = centerOriginalY - Math.floor(rf_size / 2);

        const drawX = Math.max(0, topLeftX);
        const drawY = Math.max(0, topLeftY);
        const drawW = Math.max(0, Math.min(28 - drawX, rf_size - (drawX - topLeftX)));
        const drawH = Math.max(0, Math.min(28 - drawY, rf_size - (drawY - topLeftY)));

        const rfXPct = (drawX / 28) * 100;
        const rfYPct = (drawY / 28) * 100;
        const rfWPct = (drawW / 28) * 100;
        const rfHPct = (drawH / 28) * 100;

        const panel = document.getElementById('reactiveInfoPanel');
        
        document.getElementById('hoverCoordVal').innerHTML = `
            <div class="bg-slate-50 px-3 py-1.5 rounded-md flex-1 text-center border border-slate-200">Coord: [${gridX}, ${gridY}]</div>
            <div class="bg-indigo-50 px-3 py-1.5 rounded-md flex-1 text-center border border-indigo-100 text-indigo-700">Value: <span class="font-bold">${val.toFixed(4)}</span></div>
        `;
        
        document.getElementById('rfTargetImg').src = document.getElementById('drawingCanvas').toDataURL();
        const rfHl = document.getElementById('rfTargetHighlight');
        rfHl.style.left = `${rfXPct}%`;
        rfHl.style.top = `${rfYPct}%`;
        rfHl.style.width = `${rfWPct}%`;
        rfHl.style.height = `${rfHPct}%`;
        rfHl.classList.remove('hidden');
    };
    
    wrapper.onmouseleave = () => {
        document.getElementById('intHighlight').classList.add('hidden');
        document.getElementById('hoverCoordVal').innerHTML = `<div class="text-slate-400 text-xs text-center w-full">Hover image to inspect</div>`;
        // Do not clear the rfTargetImg entirely so it doesn't abruptly disappear
        document.getElementById('rfTargetHighlight').classList.add('hidden');
    };
}
