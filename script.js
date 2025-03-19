let chart; // Store the chart object

async function trainModel() {
    // Generate random training data
    const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]); // Inputs
    const ys = tf.tensor2d([2, 4, 6, 8, 10], [5, 1]); // Outputs (y = 2x)

    // Define a simple model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // Compile the model
    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    // Train the model
    await model.fit(xs, ys, { epochs: 50 });

    // Predict new values
    const predictions = model.predict(tf.tensor2d([6, 7, 8, 9, 10], [5, 1]));
    const predArray = await predictions.data();

    // Display the result in a chart
    visualize(predArray);
}

function visualize(predictions) {
    const labels = [6, 7, 8, 9, 10]; // Input values
    const data = predictions.map(p => p.toFixed(2)); // Format data

    // Destroy previous chart if it exists
    if (chart) chart.destroy();

    // Create a new chart
    const ctx = document.getElementById("chart").getContext("2d");
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Predictions",
                data: data,
                borderColor: "blue",
                fill: false
            }]
        }
    });
}
