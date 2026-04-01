const fs = require('fs');
const path = require('path');

async function evaluate() {
    let modelArg = 'file://docs/model/model.json';
    let samplesArg = 'scripts/test_samples.json';
    let outputArg = 'scripts/tfjs_results.json';

    for (let i = 2; i < process.argv.length; i++) {
        if (process.argv[i] === '--model' && i + 1 < process.argv.length) {
            modelArg = process.argv[++i];
        } else if (process.argv[i] === '--samples' && i + 1 < process.argv.length) {
            samplesArg = process.argv[++i];
        } else if (process.argv[i] === '--output' && i + 1 < process.argv.length) {
            outputArg = process.argv[++i];
        }
    }
    
    // Defer loading TFJS so we can display progress
    console.log(`Loading @tensorflow/tfjs-node...`);
    const tf = require('@tensorflow/tfjs-node');

    console.log(`Loading test samples from ${samplesArg}...`);
    const sampleData = JSON.parse(fs.readFileSync(samplesArg, 'utf8'));
    const images = sampleData.images;
    const labels = sampleData.labels;
    const n = images.length;
    console.log(`Loaded ${n} samples.`);

    let modelPath = modelArg;
    if (!modelPath.startsWith('http') && !modelPath.startsWith('file://')) {
        const path = require('path');
        modelPath = 'file://' + path.resolve(modelPath).replace(/\\/g, '/');
    }
    
    console.log(`Loading TFJS model from ${modelPath}...`);
    const model = await tf.loadGraphModel(modelPath);

    console.log(`Evaluating predictions in batches...`);
    const flatData = [].concat(...images); // 784 * n
    const tensorData = tf.tensor4d(flatData, [n, 28, 28, 1], 'float32');
    
    const batchSize = 100;
    const predictions = [];
    const predLabels = [];

    // Memory efficient predictable loop
    for (let i = 0; i < n; i += batchSize) {
        const end = Math.min(i + batchSize, n);
        const batch = tensorData.slice([i, 0, 0, 0], [end - i, 28, 28, 1]);
        const predsTensor = model.predict(batch);
        const predsArray = await predsTensor.array();
        
        predsArray.forEach(p => {
            predictions.push(p);
            let maxIdx = 0;
            let maxVal = -1;
            p.forEach((val, idx) => {
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = idx;
                }
            });
            predLabels.push(maxIdx);
        });
        
        batch.dispose();
        predsTensor.dispose();
    }
    
    let correct = 0;
    for (let i = 0; i < n; i++) {
        if (predLabels[i] === labels[i]) {
            correct++;
        }
    }
    const accuracy = correct / n;
    console.log(`TFJS Accuracy evaluated on ${n} samples: ${(accuracy*100).toFixed(2)}%`);
    
    const results = {
        accuracy: accuracy,
        predicted_labels: predLabels,
        confidences: predictions
    };
    
    fs.writeFileSync(outputArg, JSON.stringify(results));
    console.log(`Saved results to ${outputArg}`);
}

evaluate().catch(err => {
    console.error(err);
    process.exit(1);
});
